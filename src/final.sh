#!/usr/bin/env bash

set -e 

# Check for output directory, and create it if missing
final="../style_output"
if [ ! -d "$final" ]; then
  mkdir $final
fi


main(){
  input_dir=$1
  style_dir=$2
  for cg in `ls $input_dir | egrep ".png|.PNG|.jpg|.JPG"`
  do
    content_name="${cg%.*}"
    for sg in `ls $style_dir | egrep ".png|.PNG|.jpg|.JPG"`
    do
      style_name="${sg%.*}"
      start=`date +%s`
      echo "style transfer content $cg with style $sg"
      python3 -u style_transfer.py --content $input_dir/$cg --style $style_dir/$sg  --output $final/"$content_name"_"$style_name"".png"
      end=`date +%s`
      echo "cost time is $((end-start))""s"
    done
  done
           
}



# $1 content image dir,$2 style image dir
main $1 $2
