import copy
from collections import abc
from torch import optim

class Fp16Optimizer():
  def __init__(self,init_optimizer,loss_scale=1.0,fp16=True):
    self.optimizer=init_optimizer
    self.fp16=fp16
    self.fp16_param_groups=[]
    self._fp32_param_groups = []
    self.loss_scale=loss_scale
    self.first_closure_call_this_step = True
    self.before_run()

  def before_run(self):
    if self.fp16:
      # fp16
      for i, param_group in enumerate(self.optimizer.param_groups):
          self.fp16_param_groups.append(param_group['params'])
      # seperate
      self.optimizer.param_groups = copy.deepcopy(self.optimizer.param_groups)
      # fp32
      for i, param_group in enumerate(self.optimizer.param_groups):
        self.fp32_param_groups.append(param_group['params'])
      # half fp16
      for param_group in self.fp16_param_groups:
        for param in param_group:
          param.data = param.data.half()
          if param.grad is not None:
            param.grad.data = param.grad.data.half()
    else:
      self.loss_scale = 1.0

  @property
  def fp32_param_groups(self):
    return self._fp32_param_groups

  @fp32_param_groups.setter
  def fp32_param_groups(self,param_groups):
    if not isinstance(param_groups, abc.Iterable):
      raise TypeError("params argument given to Fp16Optimizer should be "
                      "an iterable of Tensors , but got {} ".format(type(param_groups)))
    self._fp32_param_groups=[]
    for i, param_group in enumerate(param_groups):
      self._fp32_param_groups.append(param_group)



  def update_grads_to_fp32(self):
    """Update gradients from fp16 model to fp32 weight copy."""
    if self.fp16:
      for fp32_group, fp16_group in zip( self.fp32_param_groups, self.fp16_param_groups):
        for fp32_param,fp16_param in zip(fp32_group,fp16_group):
          if fp16_param.grad is not None:
            if fp32_param.grad is None:
              fp32_param.grad = fp32_param.data.new(fp32_param.size())
            fp32_param.grad.copy_(fp16_param.grad)
            if self.loss_scale != 1.0:
              fp32_param.grad.div_(self.loss_scale)




  def copy_params_to_fp16(self):
    """Copy updated params from fp32 weight copy to fp16 model."""
    for fp32_group, fp16_group in zip(self.fp32_param_groups, self.fp16_param_groups):
      for fp32_param, fp16_param in zip(fp32_group, fp16_group):
        fp16_param.data.copy_(fp32_param.data)


  def step(self,closure=None):
    if closure is not None:
      retval = self._step_with_closure(closure)
    else:
      retval = self.optimizer.step()

    self.copy_params_to_fp16()

    return retval

  def _step_with_closure(self, closure):
    def wrapped_closure():
      if self.first_closure_call_this_step:
        self.first_closure_call_this_step = False
      else:
        self.copy_params_to_fp16()
      loss = closure()
      return loss

    retval = self.optimizer.step(wrapped_closure)
    # flat_grad = self.optimizer._gather_flat_grad()
    # abs_grad_sum = flat_grad.abs().sum()
    # print (abs_grad_sum)
    self.first_closure_call_this_step = True
    return retval

  def backward(self, loss, update_master_grads=True, retain_graph=False):
    scaled_loss = loss.float() * self.loss_scale
    scaled_loss.backward(retain_graph=retain_graph)
    if update_master_grads:
      self.update_grads_to_fp32()

  def zero_grad(self):
    """
    Zero fp32 and fp16 parameter grads.
    """
    self.optimizer.zero_grad()
    # Zero fp16 gradients :
    for fp16_group in self.fp16_param_groups:
      for param in fp16_group:
        if param.grad is not None:
          param.grad.detach_()
          param.grad.zero_()




