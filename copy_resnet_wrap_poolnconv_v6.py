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
from heatmaphelpers import *

from lrp_general6 import *






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
        self.elt=sum_stacked2() # eltwisesum2()

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

        out = self.elt( torch.stack([out,identity], dim=0) ) #self.elt(out,identity)
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
        self.elt=sum_stacked2() # eltwisesum2()


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
        out = self.elt( torch.stack([out,identity], dim=0) ) #self.elt(out,identity)
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

    def copyfromresnet(self,net, lrp_params, lrp_layer2method):
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
        print('at src_module_name', src_module_name )

        foundsth=False


        if isinstance(src_module, nn.Linear):
          #copy linear layers
          foundsth=True
          print('is Linear')
          #m =  oneparam_wrapper_class( copy.deepcopy(src_module) , linearlayer_eps_wrapper_fct(), parameter1 = linear_eps )
          wrapped = get_lrpwrapperformodule( copy.deepcopy(src_module) , lrp_params, lrp_layer2method)
          if False== self.setbyname(src_module_name, wrapped ):
            raise Modulenotfounderror("could not find module "+src_module_name+ " in target net to copy" )
          updated_layers_names.append(src_module_name)
        # end of if


        if isinstance(src_module, nn.Conv2d):
          #store conv2d layers
          foundsth=True
          print('is Conv2d')
          last_src_module_name=src_module_name
          last_src_module=src_module
        # end of if

        if isinstance(src_module, nn.BatchNorm2d):
          # conv-bn chain
          foundsth=True
          print('is BatchNorm2d')

          if (True == lrp_params['use_zbeta']) and (last_src_module_name == 'conv1'):
            thisis_inputconv_andiwant_zbeta = True
          else:
            thisis_inputconv_andiwant_zbeta = False

          m = copy.deepcopy(last_src_module)
          m = bnafterconv_overwrite_intoconv(m , bn = src_module)
          # wrap conv
          wrapped = get_lrpwrapperformodule( m , lrp_params, lrp_layer2method, thisis_inputconv_andiwant_zbeta = thisis_inputconv_andiwant_zbeta )

          if False== self.setbyname(last_src_module_name, wrapped  ):
            raise Modulenotfounderror("could not find module "+nametofind+ " in target net to copy" )
        
          updated_layers_names.append(last_src_module_name)
          
          # wrap batchnorm  
          wrapped = get_lrpwrapperformodule( resetbn(src_module) , lrp_params, lrp_layer2method)
          if False== self.setbyname(src_module_name, wrapped ):
            raise Modulenotfounderror("could not find module "+src_module_name+ " in target net to copy" )            
          updated_layers_names.append(src_module_name)
        # end of if


        #if False== foundsth:
        #  print('!untreated layer')
        print('\n')
      
      # sum_stacked2 is present only in the targetclass, so must iterate here
      for target_module_name, target_module in self.named_modules():

        if isinstance(target_module, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
          wrapped = get_lrpwrapperformodule( target_module , lrp_params, lrp_layer2method)

          if False== self.setbyname(target_module_name, wrapped ):
            raise Modulenotfounderror("could not find module "+src_module_name+ " in target net to copy" )            
          updated_layers_names.append(target_module_name)


        if isinstance(target_module, sum_stacked2 ):

          wrapped =  get_lrpwrapperformodule( target_module , lrp_params, lrp_layer2method)
          if False== self.setbyname(target_module_name, wrapped ):
            raise Modulenotfounderror("could not find module "+target_module_name+ " in target net , impossible!" )            
          updated_layers_names.append(target_module_name)
      
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


def test_model5(dataloader, dataset_size, model, device):

  model.train(False)



  for data in dataloader:
    # get the inputs
    #inputs, labels, filenames = data
    inputs=data['image']
    labels=data['label']    
    fns=data['filename']  

    inputs = inputs.to(device).clone()
    labels = labels.to(device)

    inputs.requires_grad=True

    print(inputs.requires_grad)
    with torch.enable_grad():
      outputs = model(inputs)

    vals,cls = torch.max(outputs, dim=1)
    outputs[0,cls].backward()

    print(inputs.grad.shape)
    rel=inputs.grad.data
    print( torch.max(rel), torch.mean(rel) )

    clsss=get_classes()


    with torch.no_grad():

      print('shp ', outputs.shape)
      vals,cls = torch.max(outputs, dim=1)
      m=torch.mean(outputs)
      print(  vals.item(), clsss[cls.item()], m.item() )
      print(fns)

    imshow2(rel.to('cpu'),imgtensor = inputs.to('cpu'))


def runstuff(skip):


  #transforms
  data_transform = transforms.Compose([

          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) # if you do five crop, then you must change this part here, as it cannot be applied to 4 tensors

      ])

  root_dir='./img'

  dset= dataset_imagenetvalpart_nolabels(root_dir, maxnum=1, skip=skip, transform=data_transform)
  dataset_size=len(dset)
  dataloader =  torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False) #, num_workers=1) 


  device=torch.device('cpu')
  #device=torch.device('cuda')


  modeltrained = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
  model = resnet18_canonized(pretrained=False)
    
  #modeltrained = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
  #model = resnet50_canonized(pretrained=False)


  lrp_params_def1={
    'conv2d_ignorebias': True, 
    'eltwise_eps': 1e-6,
    'linear_eps': 1e-6,
    'pooling_eps': 1e-6,
    'use_zbeta': False ,
  }

  lrp_layer2method={
    'nn.ReLU':          relu_wrapper_fct,
    'nn.BatchNorm2d':   relu_wrapper_fct,
    'nn.Conv2d':        conv2d_beta0_wrapper_fct,
    'nn.Linear':        linearlayer_eps_wrapper_fct,  
    'nn.AdaptiveAvgPool2d': adaptiveavgpool2d_wrapper_fct,
    'nn.MaxPool2d': maxpool2d_wrapper_fct,
    'sum_stacked2': eltwisesum_stacked2_eps_wrapper_fct,
  }


  model.copyfromresnet(modeltrained, lrp_params=lrp_params_def1, lrp_layer2method = lrp_layer2method)
  model = model.to(device)

  test_model3(dataloader, dataset_size, modeltrained, device=device)
  test_model3(dataloader, dataset_size, model, device=device)



  test_model5(dataloader, dataset_size, model, device=device)

if __name__=='__main__':

  runstuff(skip=21) # 16,20,24,21 # 68,81,90,30,99,126 (93)


