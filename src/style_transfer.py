import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from scipy.misc import imread, imsave,imresize
from torchvision import transforms
from argparse import ArgumentParser
from networks import *
from fp16Optimizer import Fp16Optimizer
# from apex.fp16_utils import FP16_Optimizer

def build_parser():
  parser = ArgumentParser()

  parser.add_argument('--content', type=str,
                      dest='content', help='content image path',
                      metavar='CONTENT', required=True)

  parser.add_argument('--style', type=str,
                      dest='style', help='style image path',
                      metavar='STYLE', required=True)

  parser.add_argument('--output', type=str,
                      dest='output', help='output image path',
                      metavar='OUTPUT', required=True)

  parser.add_argument('--fp16_mode',type=bool, help='mixed precision training',default=True)

  return parser


def transform():
  # pre and post processing for images
  prep = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                         std=[1, 1, 1]),
    transforms.Lambda(lambda x: x.mul_(255)),
  ])
  postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                               transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                    std=[1, 1, 1]),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                               ])
  postpb = transforms.Compose([transforms.ToPILImage()])
  return prep,postpa,postpb


def postp(tensor,postpa,postpb):  # to clip results in the range [0,1]
  t = postpa(tensor)
  t[t > 1] = 1
  t[t < 0] = 0
  img = postpb(t)
  return img


# def cut_image(image, cut_num, width_range, pad_size):
#   images = list()
#   for i in range(cut_num):
#     sub_image = image[ :, width_range[i][0]:width_range[i][1] + pad_size * 2,:]
#     # tmp = np.reshape(sub_image, (1,) + sub_image.shape)
#     images.append(Image.fromarray(sub_image.astype('uint8')))
#   return images


def pad_image(image, height, width):
  unit_size = 64
  pad_height = height + (unit_size - height % unit_size) + unit_size
  pad_width = width + (unit_size - width % unit_size) + unit_size
  print(pad_height, pad_width)

  pad_t_size = (pad_height - height) // 2
  pad_b_size = pad_height - height - pad_t_size
  pad_l_size = (pad_width - width) // 2
  pad_r_size = pad_width - width - pad_l_size

  pad_t = image[height - pad_t_size:, :, :]
  pad_b = image[:pad_b_size, :, :]
  image = np.concatenate([pad_t, image, pad_b], 0)

  pad_l = image[:, width - pad_l_size:, :]
  pad_r = image[:, :pad_r_size, :]
  image = np.concatenate([pad_l, image, pad_r], 1)

  return image


def unpad_image(image, org_height, org_width):
  height, width, channel = image.shape
  pad_t_size = (height - org_height) // 2
  pad_l_size = (width - org_width) // 2

  image = image[pad_t_size:pad_t_size + org_height, :, :]
  image = image[:, pad_l_size:pad_l_size + org_width, :]

  return image

def scale_img(img,max_dim=2000):
  h,w,_=img.shape
  scale=max_dim/max(h,w)
  scale=scale if scale<1 else 1
  img=imresize(img,(int(h*scale),int(w*scale)))
  return img


def stylize():
  model_dir = os.path.abspath(os.path.dirname(os.getcwd()))+'/Models/'
  parser = build_parser()
  options = parser.parse_args()
  fp16_mode = options.fp16_mode

  show_iter = 50
  level = 3    #3
  max_iter = [300, 200, 200]  ## low_dim ... high_dim
  vgg = VGG(fp16_mode=fp16_mode)
  grammatrix = GramMatrix()
  grammseloss = GramMSELoss()
  mseloss = nn.MSELoss()
  vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
  for param in vgg.parameters():
    param.requires_grad = False
  prep, postpa, postpb=transform()

  content_image = imread(options.content, mode='RGBA')
  content_image = scale_img(content_image)
  height, width, _ = content_image.shape
  c_image = content_image[:, :, :3]
  alpha = content_image[:, :, 3]
  alpha = alpha[..., np.newaxis]
  # preprocess large content images(padding and division)
  # c_image = pad_image(c_image, height, width)
  content_image = prep(c_image)
  content_image = content_image.unsqueeze(0)
  opt_img = content_image.clone()  # .clone().detach()

  style_image = imread(options.style, mode='RGB')
  style_image = scale_img(style_image)
  style_image = prep(style_image)
  style_image = style_image.unsqueeze(0)

  if torch.cuda.is_available():
    vgg = vgg.cuda()
    grammatrix = grammatrix.cuda()
    grammseloss = grammseloss.cuda()
    mseloss = mseloss.cuda()
    content_image = content_image.cuda()
    style_image = style_image.cuda()
    opt_img = opt_img.cuda()
  if fp16_mode:
    vgg.half()
    loss_scale = 0.01

  _, _, content_h, content_w = content_image.shape
  content_down_h, content_down_w = content_h // 2 ** (level - 1), content_w // 2 ** (level - 1)
  opt_img = F.interpolate(opt_img, size=(content_down_h, content_down_w))
  temp_content_image = F.interpolate(content_image, size=(content_down_h, content_down_w))

  _, _, style_h, style_w = style_image.shape
  style_down_h, style_down_w = style_h // 2 ** (level - 1), style_w // 2 ** (level - 1)
  temp_style_image = F.interpolate(style_image, size=(style_down_h, style_down_w))

  # define layers, loss functions, weights and compute optimization targets
  style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
  content_layers = ['r42']
  loss_layers = style_layers + content_layers
  loss_fns = [grammseloss] * len(style_layers) + [mseloss] * len(content_layers)

  if torch.cuda.is_available():
    loss_fns = [loss_fn.cuda() for a, loss_fn in enumerate(loss_fns)]

  # these are good weights settings:
  style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
  content_weights = [1e0]

  weights = style_weights + content_weights
  # style_targets = [grammatrix(A.float()).detach() for A in vgg(style_image , style_layers)]

  for i in range(level - 1, -1, -1):
    _max_iter = max_iter[level - 1 - i]
    if i == 0:
      opt_img = F.interpolate(opt_img, size=(content_h, content_w))
      temp_content_image = content_image
      temp_style_image = style_image
    elif i != level - 1:
      opt_img = F.interpolate(opt_img, scale_factor=2)
      _, _, content_down_h, content_down_w = opt_img.shape
      temp_content_image = F.interpolate(content_image, size=(content_down_h, content_down_w))
      style_down_h, style_down_w = style_h // 2 ** i, style_w // 2 ** i
      temp_style_image = F.interpolate(style_image, size=(style_down_h, style_down_w))

    style_targets = [grammatrix(A).detach() for A in vgg(temp_style_image, style_layers)]
    opt_img=opt_img.requires_grad_(True)  ########################
    content_targets = [A.detach() for A in vgg(temp_content_image, content_layers)]
    targets = style_targets + content_targets

    optimizer = optim.LBFGS([opt_img], history_size=10)
    if fp16_mode:
      optimizer = optim.LBFGS([opt_img], lr=0.00001, history_size=10)  ###0.00001,0.000001
      optimizer = Fp16Optimizer(optimizer, loss_scale=loss_scale,fp16=False)
      #optimizer=FP16_Optimizer(optimizer,loss_scale)
    n_iter = [0]
    while n_iter[0] <= _max_iter:
      def closure():
        optimizer.zero_grad()
        out = vgg(opt_img, loss_layers)
        layer_losses = [weights[a] * loss_fns[a](A.cuda(), targets[a].cuda()) for a, A in enumerate(out)]
        loss = sum(layer_losses)
        if fp16_mode:
          optimizer.backward(loss)
        else:
          loss.backward()
        #print (n_iter[0]," opt grad ",opt_img.grad)
        n_iter[0] += 1
        if n_iter[0] % show_iter == (show_iter - 1):
          print('Iteration: {}, loss: {}'.format(n_iter[0] + 1, loss.item()))
        del out, layer_losses
        torch.cuda.empty_cache()
        return loss
      optimizer.step(closure)
      if fp16_mode and n_iter[0]>=10:
        optimizer.optimizer.param_groups[0]['lr']=1
    # if fp16_mode:
    #   opt_img = optimizer.fp32_param_groups[0][0].float().data.clone()
    opt_img=opt_img.float().data.clone()
    out_img = postp(opt_img[0].cpu().squeeze().float(),postpa,postpb)
    output = np.array(out_img)
    imsave("../style_output/" + "{}.jpg".format(i), output)

  # save result
  out_img = postp(opt_img[0].cpu().squeeze().float(),postpa,postpb)
  output = np.array(out_img)
  # output = unpad_image(out_img, height, width)
  output = np.concatenate([output, alpha], 2)
  imsave(options.output, output)

if __name__=='__main__':
  stylize()

