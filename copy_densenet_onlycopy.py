
from __future__ import print_function, division


import torch

import torchvision
from torchvision import datasets, models, transforms



from torch.utils.data import  DataLoader
from torchvision import transforms, utils



from getimagenetclasses import *
from dataset_imagenet2500 import dataset_imagenetvalpart_nolabels


import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import copy

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from torchvision.models import DenseNet

#############

from lrp_general6 import *


class  Modulenotfounderror(Exception):
  pass



class densenet_x(models.DenseNet):

  def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):

    super(densenet_x, self).__init__(growth_rate, block_config,num_init_features, bn_size, drop_rate, num_classes, memory_efficient)

    self.toprelu=nn.ReLU()
    self.toppool= nn.AdaptiveAvgPool2d([1,1]) #nn.AvgPool2d((7,7))

  def forward(self, x):


    features = self.features(x)
    #print('mn',torch.mean(features))


    out = self.toprelu(features)
    #print('mn',features.shape)

    out = self.toppool(out)
    print('mn2',out.shape)
    out = torch.flatten(out, 1)

    #exit()
    out = self.classifier(out)
    return out


  def setbyname(self,name,value):

    ##
    def iteratset(obj,components,value):
      #print('components',components)
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
    ##

    components=name.split('.')
    success=iteratset(self,components,value)
    return success


##############
  def copyfromdensenet(self,net):
    assert(isinstance(net, models.DenseNet) )

    name_prev2= None
    mod_prev2= None

    name_prev1= None
    mod_prev1= None


    updated_layers_names=[]
    for name,mod in net.named_modules():
    
      print('curchain:', name_prev2, name_prev1, name)
  
      # treat the first conv in the NN and its subsequent BN layer
      if name=='features.norm0': # fuse first conv with subsequent BatchNorm layer
        print('trying to update ', 'features.norm0' , 'features.conv0')
        if  name_prev1 != 'features.conv0':
          raise Modulenotfounderror( 'name_prev1 expected to be features.conv0, but found:'+name_prev1)
        #
        conv = copy.deepcopy(mod_prev1)
        conv = bnafterconv_overwrite_intoconv(conv,bn = mod)
        success = self.setbyname(name= 'features.conv0' ,value = conv)

        if False==success:
          raise Modulenotfounderror( ' could not find ','features.conv0' )

        bn = resetbn( copy.deepcopy(mod) )
        success = self.setbyname(name= 'features.norm0' ,value = bn)

        if False==success:
          raise Modulenotfounderror( ' could not find ','features.norm0' )

        updated_layers_names.append('features.conv0')
        updated_layers_names.append('features.norm0')

      elif name == 'classifier': # fuse densenet head, which has a structure 
                                #BN(norm5)-relu(toprelu)-adaptiveAvgPool(toppool)-linear
        print('trying to update ', 'classifier' , 'features.norm5','toprelu')
        

        if  name_prev1 != 'features.norm5':
          #if that fails, run an inner loop to get 'features.norm5'
          raise Modulenotfounderror( 'name_prev1 expected to be features.norm5, but found:'+name_prev1)
       
        # approach:
        #    BN(norm5)-relu(toprelu)-adaptiveAvgPool(toppool)-linear('classifier')
        # = threshrelu - BN - adaptiveAvgPool(toppool)-linear
        # = threshrelu - adaptiveAvgPool(toppool) - BN  -linear # yes this should commute bcs of no zero padding!
        # = threshrelu - adaptiveAvgPool(toppool) - fusedlinear with tensorbias
        # = resetbn(BN) -  threshrelu/clamplayer(toprelu) -  adaptiveAvgPool(toppool) - fusedlinear with tensorbias
   

        #resetbn(BN)
        success = self.setbyname(name= 'features.norm5' ,value = resetbn(mod_prev1) )
        if False==success:
          raise Modulenotfounderror( ' could not find ','features.norm5' )

        #get the right threshrelu/clamplayer
        threshrelu= getclamplayer(mod_prev1)
        success = self.setbyname(name= 'toprelu' ,value = threshrelu )
        if False==success:
          raise Modulenotfounderror( ' could not find ','toprelu')

        #get the right linearlayer with tensor boas
        linearlayer_with_biastensor = linearafterbn_returntensorbiasedlinearlayer(linearlayer=mod,bn= mod_prev1)
        success = self.setbyname(name= 'classifier' , value = linearlayer_with_biastensor)
        if False==success:
          raise Modulenotfounderror( ' could not find ','features.classifier' )

        #no need to touch the pooling
        updated_layers_names.append('classifier')
        updated_layers_names.append('features.norm5')  
        updated_layers_names.append('toprelu')  


      elif 'conv' in name: # fuse general BN-relu-conv triples in densenet

        if name == 'features.conv0':

          name_prev2= name_prev1     
          mod_prev2= mod_prev1  

          name_prev1 = name
          mod_prev1 = mod   

          continue

        # approach: 
        #    BN-relu-conv
        # =  threshrelu/clamplayer-BN-conv
        # =  threshrelu/clamplayer-(fused conv with tensorbias) # the bias is tensorshaped 
        #         with difference in spatial dimensions, whenever zero padding is used!!
        # = resetbn(BN)- threshrelu/clamplayer- (fused conv with tensorbias)

        print('trying to update BN-relu-conv chain: ', name_prev2,name_prev1,name)

        # bn-relu-conv chain

        if not isinstance(mod_prev2,nn.BatchNorm2d):
          print( 'error: no bn at the start, ', name_prev2,name_prev1,name) 
          exit()
        if not isinstance(mod_prev1,nn.ReLU):
          print( 'error: no relu in the middle, ', name_prev2,name_prev1,name) 
          exit()
      
        #get the right threshrelu/clamplayer
        clampl2= getclamplayer(bn = mod_prev2 )

        success = self.setbyname(name= name_prev2 ,value = clampl2)
        if False==success:
          raise Modulenotfounderror( ' could not find ',name_prev2 )
        
        #get the right convolution, likely with tensorbias
        convm2 = convafterbn_returntensorbiasedconv(conv = mod, bn = mod_prev2 )

        success = self.setbyname(name= name ,value = convm2)
        if False==success:
          raise Modulenotfounderror( ' could not find ',name )

        #reset batchnorm
        success = self.setbyname(name= name_prev1 ,value = resetbn(copy.deepcopy(mod_prev2)))
        if False==success:
          raise Modulenotfounderror( ' could not find ',name_prev1 )
        
        updated_layers_names.append(name)
        updated_layers_names.append(name_prev1)
        updated_layers_names.append(name_prev2)



      # read
      name_prev2= name_prev1     
      mod_prev2= mod_prev1  

      name_prev1 = name
      mod_prev1 = mod   


    print('not updated ones:')
    for target_module_name, target_module in self.named_modules():
      if target_module_name not in updated_layers_names:
        print('not updated:', target_module_name)



#############3

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}
import re
def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)

def _densenet_x(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = densenet_x(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121_x(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    return _densenet_x('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)




def test_model3(dataloader, dataset_size, model, device, printmodeldetails=False):

  from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock





  if printmodeldetails:

    print(model)
    print('\n\n\n\n\n\n')
    '''
    for module_name, module in model.named_modules():
      print('module_name', module_name )#,module )

      foundsth=False

      if isinstance(module, nn.Conv2d):
        foundsth=True
        print('is Conv2d')
      if isinstance(module, nn.BatchNorm2d):
        foundsth=True
        print('is BatchNorm2d')

      if False== foundsth:
        print('!unidentified layer')
      print('\n')
    '''


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



  #model
  #modeltrained = models.densenet121(pretrained=True)
  modeltrained = densenet121_x(pretrained=True)

  device=torch.device('cpu')
  modeltrained = modeltrained.to(device)



  model = densenet121_x(pretrained=False)
  model.copyfromdensenet(modeltrained)

  #for _,mod in modeltrained.named_modules():
  #  mod.register_forward_hook(hook_fn)
  #for _,mod in model.named_modules():
  #  mod.register_forward_hook(hook_fn)

  m1, outputs1=test_model3(dataloader, dataset_size, model, device=device, printmodeldetails=False)

  m2, outputs2=test_model3(dataloader, dataset_size, modeltrained, device=device, printmodeldetails=False)

  print('\n\n m1,m2',m1,m2  )
  print('diff of means: ', m1-m2)
  print('MAE diff of logits: ',  torch.mean(torch.abs(outputs1-outputs2)).item()  )

if __name__=='__main__':

  runstuff(skip=50) # 16,20,24,21

