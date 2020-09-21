# Segment removal using the FCN-ResNet101

The [remove-bg.py](https://github.com/ChuckTG/Segment_Removal_Pytorch_FCN-ResNet/blob/master/bg_remove.py) file is a python script that removes all the parts of an image except
the regions that represent a class that we want to detect. It then saves the resulting picture with name 'removed_bg.jpg' 

## Packages Required
* numpy
* torch
* torchvision
* PIL
* matplotlib

## Instructions
to run the program:
```bash
git clone https://github.com/ChuckTG/Segment_Removal_Pytorch_FCN-ResNet.git
cd Segment_Removal_Pytorch_FCN-ResNet

python remove_bg.py
```
## Example

This is an output produced by the script:
![](examples/example.png)