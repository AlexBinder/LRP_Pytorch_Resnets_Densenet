from __future__ import print_function, division


import torch
import torch.nn as nn


import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import  DataLoader
from torchvision import transforms, utils


import copy
import numpy as np
import matplotlib.pyplot as plt


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from torchvision.models.resnet import BasicBlock,Bottleneck,ResNet

#############
from getimagenetclasses import *
from dataset_imagenet2500 import dataset_imagenetvalpart_nolabels

from lrp_general_start2 import *
#from lrp_general6 import *




##########
##########
##########
##########
##########
##########



#partial replacement of BN, use own classes, no pretrained loading


class Cannotloadmodelweightserror(Exception):
  pass

class  Modulenotfounderror(Exception):
  pass


class BasicBlock_fused(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_fused, self).__init__(inplanes, planes, stride, downsample, groups,
                 base_width, dilation, norm_layer)

        #own
        self.elt=eltwisesum2()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        #out += identity
        #out = self.relu(out)

        out = self.elt(out,identity)
        out = self.relu(out)

        return out

class Bottleneck_fused(Bottleneck):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck_fused, self).__init__(inplanes, planes, stride, downsample, groups,
                 base_width, dilation, norm_layer)
 
        #own
        self.elt=eltwisesum2()


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        #out += identity
        out = self.elt(out,identity)
        out = self.relu(out)

        return out



class ResNet_canonized(ResNet):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_canonized, self).__init__(block, layers, num_classes, zero_init_residual,
                 groups, width_per_group, replace_stride_with_dilation,
                 norm_layer)

        ######################
        # change
        ######################
        #own
        #self.avgpool = nn.AvgPool2d(kernel_size=7,stride=7 ) #nn.AdaptiveAvgPool2d((1, 1))

        

    # runs in your current module to find the object layer3.1.conv2, and replaces it by the obkect stored in value (see         success=iteratset(self,components,value) as initializer, can be modified to run in another class when replacing that self)
    def setbyname(self,name,value):

        def iteratset(obj,components,value):

          if not hasattr(obj,components[0]):
            return False
          elif len(components)==1:
            setattr(obj,components[0],value)
            #print('found!!', components[0])
            #exit()
            return True
          else:
            nextobj=getattr(obj,components[0])
            return iteratset(nextobj,components[1:],value)

        components=name.split('.')
        success=iteratset(self,components,value)
        return success

    def copyfromresnet(self,net):
      assert( isinstance(net,ResNet))


      # --copy linear
      # --copy conv2, while fusing bns
      # --reset bn

      # first conv, then bn,
      #means: when encounter bn, find the conv before -- implementation dependent


      updated_layers_names=[]

      last_src_module_name=None
      last_src_module=None

      for src_module_name, src_module in net.named_modules():
        print('at src_module_name', src_module_name )#,module )

        foundsth=False


        #copy linear layers
        if isinstance(src_module, nn.Linear):
          foundsth=True
          print('is Linear')

          if False== self.setbyname(src_module_name,copy.deepcopy(src_module)):
            raise Modulenotfounderror("could not find module "+src_module_name+ " in target net to copy" )
          updated_layers_names.append(src_module_name)
        # end of if

        #store conv2d layers
        if isinstance(src_module, nn.Conv2d):
          foundsth=True
          print('is Conv2d')
          last_src_module_name=src_module_name
          last_src_module=src_module
        # end of if
        '''
        if isinstance(src_module, nn.BatchNorm2d):
          foundsth=True
          print('is BatchNorm2d')

          #find: testmodulename = the name which the corresponding conv layer should have
          pos=src_module_name.find('bn')
          if pos != -1: #structure: .bn<number> something
            prefix=src_module_name[0:pos]
            postfix=src_module_name[pos+2:] # 2 = length('bn')
            testmodulename=prefix+'conv'+postfix
          else:
            pos=src_module_name.find('downsample')
            if pos != -1:
              testmodulename=src_module_name[:-2] +'.0'
            else:
              print('!!!!', src_module_name )
              exit()
         
          # got it
          #print(testmodulename)
          assert(testmodulename== last_src_module_name)
          #or raise
          #else find in target conv 
        '''

        if isinstance(src_module, nn.BatchNorm2d):
          foundsth=True
          print('is BatchNorm2d')
          m = copy.deepcopy(last_src_module)
          m = bnafterconv_overwrite_intoconv(m , bn = src_module)

          # copy conv
          if False== self.setbyname(last_src_module_name,m):
            raise Modulenotfounderror("could not find module "+last_src_module_name+ " in target net to copy" )            
          updated_layers_names.append(last_src_module_name)

          # copy resetted batchnorm
          if False== self.setbyname(src_module_name, resetbn(src_module) ):
            raise Modulenotfounderror("could not find module "+src_module_name+ " in target net to copy" )            
          updated_layers_names.append(src_module_name)


        # end of if

        if False== foundsth:
          print('!untreated layer')
        print('\n')
      
      
      for target_module_name, target_module in self.named_modules():
        if target_module_name not in updated_layers_names:
          print('not updated:', target_module_name)
      

def _resnet_canonized(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_canonized(block, layers, **kwargs)
    if pretrained:
        raise Cannotloadmodelweightserror("explainable nn model wrapper was never meant to load dictionary weights, load into standard model first, then instatiate this class from the standard model")
    return model


def resnet18_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized('resnet18', BasicBlock_fused, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet50_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized('resnet50', Bottleneck_fused, [3, 4, 6, 3], pretrained, progress, **kwargs)





def test_model3(dataloader, dataset_size, model, device):

  from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock

  model.train(False)
  for data in dataloader:
    # get the inputs
    #inputs, labels, filenames = data
    inputs=data['image']
    labels=data['label']    
    fns=data['filename']  

    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
      outputs = model(inputs)
      print('shp ', outputs.shape)
      m=torch.mean(outputs)
      m0=torch.mean(inputs)
      print(m.item(), m0.item() )
      print(fns)

      return m.item(), outputs

def runstuff(skip):

  use_gpu=True

  #transforms
  data_transform = transforms.Compose([

          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) # if you do five crop, then you must change this part here, as it cannot be applied to 4 tensors

      ])

  root_dir='./img'

  dset= dataset_imagenetvalpart_nolabels(root_dir, maxnum=1, transform=data_transform, skip= skip)
  dataset_size=len(dset)
  dataloader =  torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False) #, num_workers=1) 


  device=torch.device('cpu')
  #device=torch.device('cuda')

  modeltrained = models.resnet18(pretrained=True).to(device)
  model = resnet18_canonized(pretrained=False)
  #modeltrained = models.resnet50(pretrained=True).to(device)
  #model = resnet50_canonized(pretrained=False)
  print(model)
  #exit()


  model.copyfromresnet(modeltrained)
  model = model.to(device)


  m1, outputs1 = test_model3(dataloader, dataset_size, modeltrained, device=device)
  m2, outputs2 = test_model3(dataloader, dataset_size, model, device=device)

  print('\n\n m1,m2',m1,m2  )
  print('diff of means: ', m1-m2)
  print('MAE diff of logits: ',  torch.mean(torch.abs(outputs1-outputs2)).item()  )

if __name__=='__main__':

  runstuff(skip=21) #16,20, 24, 21


