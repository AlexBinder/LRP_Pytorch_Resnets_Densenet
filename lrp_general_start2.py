import torch
import torch.nn as nn

import copy



def resetbn(bn):

  assert (isinstance(bn,nn.BatchNorm2d))

  bnc=copy.deepcopy(bn)
  bnc.reset_parameters()

  return bnc

class eltwisesum2(nn.Module): # see torchray excitation backprop, using *inputs
    def __init__(self):
        super(eltwisesum2, self).__init__()
      
    def vanillaforward(self, x1,x2):
          return x1+x2

    def forward(self, x1,x2):
        return(self.vanillaforward(x1,x2))




def bnafterconv_overwrite_intoconv(conv,bn): #after visatt

    s = (bn.running_var+bn.eps)**.5
    w = bn.weight
    b = bn.bias
    m = bn.running_mean

    #print(w.shape,b.shape)
    #exit()

    conv.weight = torch.nn.Parameter(conv.weight * (w / s).reshape(-1, 1, 1, 1))

    #print( 'w/s, conv.bias', (w/s), conv.bias )

    if conv.bias is None:
      conv.bias = torch.nn.Parameter((0 - m) * (w / s) + b)
    else:
      conv.bias = torch.nn.Parameter(( conv.bias - m) * (w / s) + b)
    return conv

