import torch
import torch.nn as nn

import copy




#######################################################
#######################################################
# wrappers for autograd type modules
#######################################################
#######################################################



class zeroparam_wrapper_class(nn.Module):
  def __init__(self, module, autogradfunction):
    super(zeroparam_wrapper_class, self).__init__()
    self.module=module
    self.wrapper=autogradfunction

  def forward(self,x):
    y=self.wrapper.apply( x, self.module)
    return y

class oneparam_wrapper_class(nn.Module):
  def __init__(self, module, autogradfunction, parameter1):
    super(oneparam_wrapper_class, self).__init__()
    self.module=module
    self.wrapper=autogradfunction
    self.parameter1=parameter1

  def forward(self,x):
    y=self.wrapper.apply( x, self.module,self.parameter1)
    return y



class conv2d_zbeta_wrapper_class(nn.Module):
  def __init__(self, module, lrpignorebias,lowest  = None, highest = None  ):
    super(conv2d_zbeta_wrapper_class, self).__init__()

    if lowest is None:
      lowest=torch.min(torch.tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225]))
    if highest is None:
      highest=torch.max (torch.tensor([(1-0.485)/0.229, (1-0.456)/0.224, (1-0.406)/0.225]))
    assert( isinstance( module, nn.Conv2d ))

    self.module=module
    self.wrapper=conv2d_zbeta_wrapper_fct()

    self.lrpignorebias=lrpignorebias

    self.lowest=lowest
    self.highest=highest

  def forward(self,x):
    y=self.wrapper.apply( x, self.module, self.lrpignorebias, self.lowest, self.highest)
    return y


class  lrplookupnotfounderror(Exception):
  pass

def get_lrpwrapperformodule(module, lrp_params, lrp_layer2method, thisis_inputconv_andiwant_zbeta=False):

  if isinstance(module, nn.ReLU):
    #return zeroparam_wrapper_class( module , relu_wrapper_fct() )

    key='nn.ReLU'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default relu_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return zeroparam_wrapper_class( module , autogradfunction= autogradfunction )

  elif isinstance(module, nn.BatchNorm2d):

    key='nn.BatchNorm2d'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default relu_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return zeroparam_wrapper_class( module , autogradfunction= autogradfunction )

  elif isinstance(module, nn.Linear):

    key='nn.Linear'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default linearlayer_eps_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return oneparam_wrapper_class( module , autogradfunction = autogradfunction, parameter1 = lrp_params['linear_eps'] )

  elif isinstance(module, nn.Conv2d): 
    if True== thisis_inputconv_andiwant_zbeta:
      return conv2d_zbeta_wrapper_class(module , lrp_params['conv2d_ignorebias'])
    else:
      key='nn.Conv2d'
      if key not in lrp_layer2method:
        print("found no dictionary entry in lrp_layer2method for this module name:", key)
        raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

      #default conv2d_beta0_wrapper_fct()
      autogradfunction = lrp_layer2method[key]()
      return oneparam_wrapper_class(module, autogradfunction = autogradfunction, parameter1 = lrp_params['conv2d_ignorebias'] )

  elif isinstance(module, nn.AdaptiveAvgPool2d):

    key='nn.AdaptiveAvgPool2d'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default adaptiveavgpool2d_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return oneparam_wrapper_class( module , autogradfunction = autogradfunction, parameter1 = lrp_params['pooling_eps'] )

  elif isinstance(module, nn.AvgPool2d):

    key='nn.AvgPool2d'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default adaptiveavgpool2d_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return oneparam_wrapper_class( module , autogradfunction = autogradfunction, parameter1 = lrp_params['pooling_eps'] )
    
    
  elif isinstance(module, nn.MaxPool2d):

      key='nn.MaxPool2d'
      if key not in lrp_layer2method:
        print("found no dictionary entry in lrp_layer2method for this module name:", key)
        raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

      #default maxpool2d_wrapper_fct()
      autogradfunction = lrp_layer2method[key]()
      return zeroparam_wrapper_class( module , autogradfunction = autogradfunction )



  elif isinstance(module, sum_stacked2): # resnet specific

    key='sum_stacked2'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default eltwisesum_stacked2_eps_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return oneparam_wrapper_class( module , autogradfunction = autogradfunction, parameter1 = lrp_params['eltwise_eps'] )
  
  elif isinstance(module, clamplayer): # densenet specific

    key='clamplayer'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default relu_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return zeroparam_wrapper_class( module , autogradfunction = autogradfunction)

  elif isinstance(module, tensorbiased_linearlayer): # densenet specific 
       
    key='tensorbiased_linearlayer'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default relu_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return oneparam_wrapper_class( module , autogradfunction = autogradfunction, parameter1 = lrp_params['linear_eps'] )

  elif isinstance(module, tensorbiased_convlayer): # densenet specific 
       
    key='tensorbiased_convlayer'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default relu_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return oneparam_wrapper_class( module , autogradfunction = autogradfunction, parameter1 = lrp_params['conv2d_ignorebias'] )


  else:
    print("found no lookup for this module:", module)
    raise lrplookupnotfounderror( "found no lookup for this module:", module)

'''
#replaced by sum2 with torch.stack([x1,x2],dim=0)
# resnet specific, because it calls 2 inputs x1,x2
class eltwise2_wrapper_class(nn.Module):
  def __init__(self,eps):
    super(eltwise2_wrapper_class, self).__init__()

    self.module=eltwisesum2()
    self.wrapper=eltwisesum_eps_wrapper_fct()
    self.eps=eps

  def forward(self,x1,x2):
    y=self.wrapper.apply( x1,x2, self.module,self.eps)
    return y
'''

#######################################################
#######################################################
#canonization functions
#######################################################
#######################################################

def resetbn(bn):

  assert (isinstance(bn,nn.BatchNorm2d))

  bnc=copy.deepcopy(bn)
  bnc.reset_parameters()

  return bnc


#vanilla fusion conv-bn --> conv(updatedparams)
def bnafterconv_overwrite_intoconv(conv,bn): #after visatt

        
    print(conv,bn)

    assert (isinstance(bn,nn.BatchNorm2d))
    assert (isinstance(conv,nn.Conv2d))

    s = (bn.running_var+bn.eps)**.5
    w = bn.weight
    b = bn.bias
    m = bn.running_mean
    conv.weight = torch.nn.Parameter(conv.weight * (w / s).reshape(-1, 1, 1, 1))

    #print( 'w/s, conv.bias', (w/s), conv.bias )

    if conv.bias is None:
      conv.bias = torch.nn.Parameter((0 - m) * (w / s) + b)
    else:
      conv.bias = torch.nn.Parameter(( conv.bias - m) * (w / s) + b)
    #print( ' conv.bias new',  conv.bias )
    return conv






def getclamplayer(bn):

    assert (isinstance(bn,nn.BatchNorm2d))

    var_bn = (bn.running_var+bn.eps)**.5
    w_bn = bn.weight
    bias_bn = bn.bias
    mu_bn = bn.running_mean

    if torch.norm(w_bn) > 0:
      thresh = -bias_bn * var_bn / w_bn + mu_bn
      clamplay = clamplayer( thresh ,torch.sign(w_bn) , forconv = True  )
    else:
      print('bad case, not (torch.norm(w_bn) > 0), exiting, see lrp_general*.py, you can outcomment the exit(), but it means that your batchnorm layer is messed up.')
      exit()
      if (bias_bn  < 0): # never activated
        thresh = 0
        clamplay = clamplayer_const( thresh ,  forconv = True  )
      else:
        thresh = bias_bn     # never activated always fires the bias
        spec = True     
        clamplay = clamplayer_const( thresh ,  forconv = True  )

    return clamplay


# for fusion of bn-conv into conv with tensor shaped bias
def convafterbn_returntensorbiasedconv(conv,bn): #after visatt
    
    # y= w_{bn} (x-mu_{bn}) / vareps + b_{bn}
    # y= w_{bn} / vareps  x   -  w_{bn} / vareps * mu_{bn} + b_{bn}
    # conv(y) = w y +b =   w ( w_{bn} / vareps) x    + w (  -  w_{bn} / vareps * mu_{bn} + b_{bn}  )  +b 
    #   conv(y) =   w ( w_{bn} / vareps).reshape((-1,1,1,1)) x  + conv ( -  w_{bn} / vareps * mu_{bn} + b_{bn}  )
    # the latter as bias for  mu_{bn},  b_{bn} as ksize shaped tensors


    assert (isinstance(bn,nn.BatchNorm2d))
    assert (isinstance(conv,nn.Conv2d))

    var_bn = (bn.running_var.clone().detach()+bn.eps)**.5
    w_bn = bn.weight.clone().detach()
    bias_bn = bn.bias.clone().detach()
    mu_bn = bn.running_mean.clone().detach()



    newweight=conv.weight.clone().detach() * (w_bn / var_bn).reshape(1, conv.weight.shape[1], 1, 1)

    inputfornewbias=  - (w_bn / var_bn * mu_bn)+ bias_bn #size [nchannels]

    # can consider to always return tensorbiased ...
    if conv.padding==0:

      ksize=( conv.weight.shape[2], conv.weight.shape[3] )
      inputfornewbias2=inputfornewbias.unsqueeze(1).unsqueeze(2).repeat((1, ksize[0] , ksize[1])).unsqueeze(0) # shape (1, 64, ksize,ksize)

      #print (inputfornewbias.shape)

      with torch.no_grad():
        prebias= conv( inputfornewbias2 )
      # in case of padding, the result might be not (1,nch,1,1)
      mi = ( ( prebias.shape[2]-1)//2 , ( prebias.shape[3]-1)//2 )
      prebias= prebias.clone().detach() 
      newconv_bias=   prebias[0,:,mi[0],mi[1]] 

      conv2=copy.deepcopy(conv)
      conv2.weight =  torch.nn.Parameter( newweight )
      conv2.bias =  torch.nn.Parameter( newconv_bias)
      return conv2

    else:
      #bias is a tensor! if there is padding

      spatiallybiasedconv=  tensorbiased_convlayer(newweight, conv,inputfornewbias.clone().detach())
      return spatiallybiasedconv


# for fusion of bn-linear into linear with tensor shaped bias
def linearafterbn_returntensorbiasedlinearlayer(linearlayer,bn): #after visatt

    assert (isinstance(bn,nn.BatchNorm2d))
    assert (isinstance(linearlayer,nn.Linear))

    var_bn = (bn.running_var+bn.eps)**.5
    w_bn = bn.weight
    bias_bn = bn.bias
    mu_bn = bn.running_mean

    newweight = torch.nn.Parameter(linearlayer.weight.clone().detach() * (w_bn / var_bn))

    inputfornewbias=  - (w_bn / var_bn * mu_bn)+ bias_bn #size [nchannels]
    inputfornewbias= inputfornewbias.detach()

    with torch.no_grad():
      newbias= linearlayer.forward( inputfornewbias )
    #print(newbias.shape, newbias.numel(),linearlayer.out_features)
    #exit()
    tensorbias_linearlayer = tensorbiased_linearlayer(linearlayer.in_features, linearlayer.out_features , newweight, newbias.data)

    return tensorbias_linearlayer




#resnet stuff

###########################################################
#########################################################
###########################################################

#for resnet shortcut / residual connections
class eltwisesum2(nn.Module): # see torchray excitation backprop, using *inputs
    def __init__(self):
        super(eltwisesum2, self).__init__()

    def forward(self, x1,x2):
        return x1+x2


#densenet stuff

###########################################################
#########################################################
###########################################################

#bad name actually, threshrelu would be better
class clamplayer(nn.Module):

  def __init__(self,thresh, w_bn_sign, forconv):
    super(clamplayer,self).__init__()

    # thresh will be -b_bn*vareps / w_bn + mu_bn
    if True == forconv:
      self.thresh = thresh.reshape((-1,1,1))
      self.w_bn_sign = w_bn_sign.reshape((-1,1,1))
    else:
      # for linear that should be ok
      self.thresh = thresh
      self.w_bn_sign = w_bn_sign

  def forward(self,x):
    # for channels c with w_bn > 0  -- as checked by (self.w_bn_sign>0)
    # return (x- self.thresh ) * (x>self.thresh) +  self.thresh 
    #
    # for channels c with w_bn < 0
    # return thresh if (x>=self.thresh), x  if (x < self. thresh)
    # return (x- self.thresh ) * (x<self.thresh) +  self.thresh 
 
    return (x- self.thresh ) * ( (x>self.thresh)* (self.w_bn_sign>0)   + (x<self.thresh)* (self.w_bn_sign<0)  ) +  self.thresh 



class  tensorbiased_linearlayer(nn.Module):
  def __init__(self, in_features, out_features , newweight, newbias ):
    super(tensorbiased_linearlayer,self).__init__()
    assert( newbias.numel()== out_features )

    self.linearbase=nn.Linear(in_features, out_features, bias=False)
    self.linearbase.weight = torch.nn.Parameter(newweight)
    self.biastensor =  torch.nn.Parameter(newbias)

    # this is pure convenience
    self.in_features= in_features
    self.out_features= out_features

  def forward(self,x):
    y= self.linearbase.forward(x) + self.biastensor
    return y




class  tensorbiased_convlayer(nn.Module):

  def _clone_module(self, module):
      clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                   **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
      return clone.to(module.weight.device)

  def __init__(self, newweight, baseconv, inputfornewbias):
    super(tensorbiased_convlayer, self).__init__()

    self.baseconv=baseconv #evtl store weights, clone mod

    self.inputfornewbias=inputfornewbias

    self.conv=  self._clone_module(baseconv)
    self.conv.weight = torch.nn.Parameter(newweight)
    self.conv.bias=None

    self.biasmode='neutr'

  def gettensorbias(self,x):

    with torch.no_grad():
      tensorbias= self.baseconv(  self.inputfornewbias.unsqueeze(1).unsqueeze(2).repeat((1, x.shape[2] , x.shape[3])).unsqueeze(0)  )
    print('done tensorbias', tensorbias.shape)    
    return tensorbias

  def forward(self,x):
    #self.input=(x)
    print('tensorbias fwd', x.shape)
    if len(x.shape)!=4:
      print('bad tensor length')
      exit()  
    if self.inputfornewbias is not None:
      print ( 'self.inputfornewbias.shape',self.inputfornewbias.shape)
    else:
      print ( 'self.inputfornewbias',self.inputfornewbias)
    #z= self.conv.forward(x) 
    #print('z.shape',z.shape)
    if self.inputfornewbias is  None:
      return self.conv.forward(x) #z
    else:
      b= self.gettensorbias(x)
      if self.biasmode=='neutr':
        #z+=b 
        return self.conv.forward(x) +b
      elif self.biasmode=='pos':
        #z+= torch.clamp(b,min=0)
        return self.conv.forward(x) +torch.clamp(b,min=0) #z
      elif self.biasmode=='neg':
        #z+= torch.clamp(b,max=0)
        return self.conv.forward(x) +torch.clamp(b,max=0) #z




#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################



#######################################################
#######################################################
# autograd type modules
#######################################################
#######################################################


class conv2d_beta0_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module, lrpignorebias):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module):
            #[module.in_channels, module.out_channels, module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']}

            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= module.weight.device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= module.weight.device)     
              else:
                print('v is neither int nor tuple. unexpected')
                exit()
              values.append(v)
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module)

        if module.bias is None:
          bias=None
        else:
          bias= module.bias.data.clone()
        lrpignorebiastensor=torch.tensor([lrpignorebias], dtype=torch.bool, device= module.weight.device)
        ctx.save_for_backward(x, module.weight.data.clone(), bias, lrpignorebiastensor, *values ) # *values unpacks the list

        #print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()

        #print('conv2d custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, conv2dweight, conv2dbias, lrpignorebiastensor, *values  = ctx.saved_tensors
        #print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)

        if conv2dbias is None:
          module=nn.Conv2d( **paramsdict, bias=False )
        else:
          module=nn.Conv2d( **paramsdict, bias=True )
          module.bias= torch.nn.Parameter(conv2dbias)
                
        #print('conv2dconstr')
        module.weight= torch.nn.Parameter(conv2dweight)

        #print('conv2dconstr weights')

        pnconv = posnegconv(module, ignorebias = lrpignorebiastensor.item())


        #print('conv2d custom input_.shape', input_.shape )

        X = input_.clone().detach().requires_grad_(True)
        R= lrp_backward(_input= X , layer = pnconv , relevance_output = grad_output[0], eps0 = 1e-12, eps=0)

        #print('conv2d custom R', R.shape )
        #exit()
        return R,None, None

class conv2d_zbeta_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module, lrpignorebias, lowest, highest):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module):
            #[module.in_channels, module.out_channels, module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']}

            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= module.weight.device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= module.weight.device)     
              else:
                print('v is neither int nor tuple. unexpected')
                exit()
              values.append(v)
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module)

        if module.bias is None:
          bias=None
        else:
          bias= module.bias.data.clone()
        lrpignorebiastensor=torch.tensor([lrpignorebias], dtype=torch.bool, device= module.weight.device)
        ctx.save_for_backward(x, module.weight.data.clone(), bias, lrpignorebiastensor, lowest.to(module.weight.device), highest.to(module.weight.device), *values ) # *values unpacks the list

        #print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()

        #print('conv2d zbeta custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, conv2dweight, conv2dbias, lrpignorebiastensor, lowest_, highest_, *values  = ctx.saved_tensors
        #print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)

        if conv2dbias is None:
          module=nn.Conv2d( **paramsdict, bias=False )
        else:
          module=nn.Conv2d( **paramsdict, bias=True )
          module.bias= torch.nn.Parameter(conv2dbias)
                
        #print('conv2dconstr')
        module.weight= torch.nn.Parameter(conv2dweight)

        print('zbeta conv2dconstr weights')

        #pnconv = posnegconv(conv2dclass, ignorebias=True)
        #X = input_.clone().detach().requires_grad_(True)
        #R= lrp_backward(_input= X , layer = pnconv , relevance_output = grad_output[0], eps0 = 1e-12, eps=0)

        any_conv =  anysign_conv( module, ignorebias=lrpignorebiastensor.item())

        X = input_.clone().detach().requires_grad_(True)
        L = (lowest_ * torch.ones_like(X)).requires_grad_(True)
        H = (highest_ * torch.ones_like(X)).requires_grad_(True)

        with torch.enable_grad():
            Z = any_conv.forward(mode='justasitis', x=X) - any_conv.forward(mode='pos', x=L) - any_conv.forward(mode='neg', x=H) 
            S = safe_divide(grad_output[0].clone().detach(), Z.clone().detach(), eps0=1e-6, eps=1e-6)
            Z.backward(S)
            R = (X * X.grad + L * L.grad + H * H.grad).detach()

        print('zbeta conv2d custom R', R.shape )
        #exit()
        return R,None,None,None, None # for  (x, conv2dclass,lrpignorebias, lowest, highest)




class adaptiveavgpool2d_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module, eps):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module,device):

            propertynames=['output_size']
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= device)     
              else:
                print('v is neither int nor tuple. unexpected')
                exit()
              values.append(v)
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module,x.device)
        epstensor = torch.tensor([eps], dtype=torch.float32, device= x.device) 
        ctx.save_for_backward(x, epstensor, *values ) # *values unpacks the list

        print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()
        print('sdaptiveavg2d custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, epstensor, *values  = ctx.saved_tensors
        print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['output_size']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)
        eps=epstensor.item()

        #class instantiation
        layerclass= torch.nn.AdaptiveAvgPool2d(**paramsdict)

        print('adaptiveavg2d custom input_.shape', input_.shape )

        X = input_.clone().detach().requires_grad_(True)
        R= lrp_backward(_input= X , layer = layerclass , relevance_output = grad_output[0], eps0 = eps, eps=eps)

        print('adaptiveavg2dcustom R', R.shape )
        #exit()
        return R,None,None




class maxpool2d_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module,device):

            propertynames=['kernel_size','stride','padding','dilation','return_indices','ceil_mode']
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, bool):
                v=  torch.tensor([v], dtype=torch.bool, device= device) 
              elif isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= device) 
              elif isinstance(v, bool):
  
                v=  torch.tensor([v], dtype=torch.int32, device= device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= device)     
              else:
                print('v is neither int nor tuple. unexpected')
                exit()
              values.append(v)
            #exit()
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module,x.device)
        ctx.save_for_backward(x, *values ) # *values unpacks the list

        print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()
        print('maxpool2d custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, *values  = ctx.saved_tensors
        print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['kernel_size','stride','padding','dilation','return_indices','ceil_mode']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)

        layerclass= torch.nn.MaxPool2d(**paramsdict)

        X = input_.clone().detach().requires_grad_(True)
        #R= lrp_backward(_input= X , layer = layerclass , relevance_output = grad_output[0], eps0 = 1e-12, eps=0)    
        with torch.enable_grad():
            Z = layerclass.forward(X)
        relevance_output_data = grad_output[0].clone().detach().unsqueeze(0)
        Z.backward(relevance_output_data)
        R = X.grad
        
        print('maxpool R', R.shape )
        #exit()
        return R,None

class avgpool2d_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module,eps):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module,device):

            propertynames=['kernel_size','stride','padding','ceil_mode','count_include_pad','divisor_override']
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, bool):
                v=  torch.tensor([v], dtype=torch.bool, device= device) 
              elif isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= device)  
              elif v is None:
                pass   
              else:
                print('v is neither int nor tuple. unexpected', attr,type(v))
                exit()
              values.append(v)
            #exit()
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module,x.device)
        epstensor=torch.tensor([eps], dtype=torch.float32, device= x.device)  
        ctx.save_for_backward(x, epstensor, *values ) # *values unpacks the list

        print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()
        print('maxpool2d custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, epstensor, *values  = ctx.saved_tensors
        print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['kernel_size','stride','padding','ceil_mode','count_include_pad','divisor_override']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v is None:
                  paramsdict[n]=v
              elif v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)

        layerclass= torch.nn.AvgPool2d(**paramsdict)
        eps=epstensor.item()
        X = input_.clone().detach().requires_grad_(True)
        R= lrp_backward(_input= X , layer = layerclass , relevance_output = grad_output[0], eps0 = eps, eps=eps)    

        
        print('avgpool R', R.shape )
        #exit()
        return R,None,None




class relu_wrapper_fct(torch.autograd.Function): # to be used with generic_activation_pool_wrapper_class(module,this)
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module):
        def configvalues_totensorlist(module):
            propertynames=[]
            values=[]
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################
        #stash module config params and trainable params
        #propertynames,values=configvalues_totensorlist(conv2dclass)
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)
        #input_, conv2dweight, conv2dbias, *values  = ctx.saved_tensors
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=[]
            paramsdict={}
            return paramsdict
        #######################################################################
        #paramsdict=tensorlist_todict(values)
        return grad_output,None


#lineareps_wrapper_fct
class linearlayer_eps_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module, eps):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module):
            #[module.in_channels, module.out_channels, module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']}

            propertynames=['in_features','out_features']
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= module.weight.device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= module.weight.device)     
              else:
                print('v is neither int nor tuple. unexpected')
                exit()
              values.append(v)
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module)
        epstensor= torch.tensor([eps], dtype=torch.float32, device= x.device) 

        if module.bias is None:
          bias=None
        else:
          bias= module.bias.data.clone()
        ctx.save_for_backward(x, module.weight.data.clone(), bias, epstensor, *values ) # *values unpacks the list

        print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()

        print('linear custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, weight, bias, epstensor, *values  = ctx.saved_tensors
        print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['in_features','out_features']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)

        if bias is None:
          module=nn.Linear( **paramsdict, bias=False )
        else:
          module=nn.Linear( **paramsdict, bias=True )
          module.bias= torch.nn.Parameter(bias)
                
        #print('conv2dconstr')
        module.weight= torch.nn.Parameter(weight)

        print('linaer custom input_.shape', input_.shape )
        eps=epstensor.item()
        X = input_.clone().detach().requires_grad_(True)
        R= lrp_backward(_input= X , layer = module , relevance_output = grad_output[0], eps0 = eps, eps=eps)

        print('linaer custom R', R.shape )
        #exit()
        return R,None,None

class sum_stacked2(nn.Module):
    def __init__(self):
        super(sum_stacked2, self).__init__()

    @staticmethod
    def forward(x): # from X=torch.stack([X0, X1], dim=0)
        assert( x.shape[0]==2 )    
        return torch.sum(x,dim=0)




class eltwisesum_stacked2_eps_wrapper_fct(torch.autograd.Function): # to be used with generic_activation_pool_wrapper_class(module,this)
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, stackedx, module,eps):
        def configvalues_totensorlist(module):
            propertynames=[]
            values=[]
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################
        #stash module config params and trainable params
        #propertynames,values=configvalues_totensorlist(conv2dclass)

        epstensor= torch.tensor([eps], dtype=torch.float32, device= stackedx.device) 
        ctx.save_for_backward(stackedx, epstensor )
        return module.forward(stackedx)

    @staticmethod
    def backward(ctx, grad_output):
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)
        stackedx,epstensor  = ctx.saved_tensors
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=[]
            paramsdict={}
            return paramsdict
        #######################################################################
        #paramsdict=tensorlist_todict(values)

        #X0 = x1.clone().detach() #.requires_grad_(True)
        #X1 = x2.clone().detach() #.requires_grad_(True)
        #X=torch.stack([X0, X1], dim=0) # along a new dimension!

        X = stackedx.clone().detach().requires_grad_(True)

        eps=epstensor.item()

        s2=sum_stacked2().to(X.device)
        Rtmp= lrp_backward(_input= X , layer = s2 , relevance_output = grad_output[0], eps0 = eps, eps=eps)

        #R0=Rtmp[0,:]
        #R1=Rtmp[1,:]

        return Rtmp,None,None



'''
class eltwisesum_eps_wrapper_fct(torch.autograd.Function): # to be used with generic_activation_pool_wrapper_class(module,this)
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x1,x2, module,eps):
        def configvalues_totensorlist(module):
            propertynames=[]
            values=[]
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################
        #stash module config params and trainable params
        #propertynames,values=configvalues_totensorlist(conv2dclass)

        epstensor= torch.tensor([eps], dtype=torch.float32, device= x1.device) 
        ctx.save_for_backward(x1,x2, epstensor )
        return module.forward(x1,x2)

    @staticmethod
    def backward(ctx, grad_output):
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)
        x1,x2,epstensor  = ctx.saved_tensors
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=[]
            paramsdict={}
            return paramsdict
        #######################################################################
        #paramsdict=tensorlist_todict(values)

        X0 = x1.clone().detach() #.requires_grad_(True)
        X1 = x2.clone().detach() #.requires_grad_(True)

        X=torch.stack([X0, X1], dim=0) # along a new dimension!
        X.requires_grad_(True)

        eps=epstensor.item()

        s2=sum_stacked2().to(X0.device)

        Rtmp= lrp_backward(_input= X , layer = s2 , relevance_output = grad_output[0], eps0 = eps, eps=eps)

        R0=Rtmp[0,:]
        R1=Rtmp[1,:]

        return R0,R1,None,None
'''




#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################



#######################################################
#######################################################
# aux input classes
#######################################################
#######################################################

class posnegconv(nn.Module):


    def _clone_module(self, module):
        clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                     **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
      super(posnegconv, self).__init__()

      self.posconv=self._clone_module(conv)
      self.posconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(min=0) ).to(conv.weight.device)

      self.negconv=self._clone_module(conv)
      self.negconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(max=0) ).to(conv.weight.device)


      #ignbias=True
      #ignorebias=False
      if ignorebias==True:
        self.posconv.bias=None
        self.negconv.bias=None
      else:
          if conv.bias is not None:
              self.posconv.bias= torch.nn.Parameter( conv.bias.data.clone().clamp(min=0) )
              self.negconv.bias= torch.nn.Parameter( conv.bias.data.clone().clamp(max=0) )

      #print('done init')

    def forward(self,x):
        vp= self.posconv ( torch.clamp(x,min=0)  )
        vn= self.negconv ( torch.clamp(x,max=0)  )
        return vp+vn



##### input


class anysign_conv(nn.Module):


    def _clone_module(self, module):
        clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                     **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
      super( anysign_conv, self).__init__()

      self.posconv=self._clone_module(conv)
      self.posconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(min=0) ).to(conv.weight.device)

      self.negconv=self._clone_module(conv)
      self.negconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(max=0) ).to(conv.weight.device)

      self.jusconv=self._clone_module(conv)
      self.jusconv.weight= torch.nn.Parameter( conv.weight.data.clone() ).to(conv.weight.device)

      if ignorebias==True:
        self.posconv.bias=None
        self.negconv.bias=None
        self.jusconv.bias=None
      else:
          if conv.bias is not None:
              self.posconv.bias= torch.nn.Parameter( conv.bias.data.clone().clamp(min=0) ).to(conv.weight.device)
              self.negconv.bias= torch.nn.Parameter( conv.bias.data.clone().clamp(max=0) ).to(conv.weight.device)
              self.jusconv.bias= torch.nn.Parameter( conv.bias.data.clone() ).to(conv.weight.device)

      print('done init')

    def forward(self,mode,x):
        if mode == 'pos':
            return self.posconv.forward(x)
        elif mode =='neg':
            return self.negconv.forward(x)
        elif mode =='justasitis':
            return self.jusconv.forward(x)
        else:
            raise NotImplementedError("anysign_conv notimpl mode: "+ str(mode))
        return vp+vn

########################################################
#########################################################
#########################################################
###########################################################
#########################################################
###########################################################

#densenet stuff

###########################################################
#########################################################
###########################################################




class tensorbiased_linearlayer_eps_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module, eps):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module):
            #[module.in_channels, module.out_channels, module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']}

            propertynames=['in_features','out_features']
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= x.device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= x.device)     
              else:
                print('v is neither int nor tuple. unexpected')
                exit()
              values.append(v)
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module)
        epstensor= torch.tensor([eps], dtype=torch.float32, device= x.device) 

        #if module.bias is None:
        #  bias=None
        #else:
        #  bias= module.bias.data.clone()
        #ctx.save_for_backward(x, module.weight.data.clone(), bias, epstensor, *values ) # *values unpacks the list
        ctx.save_for_backward(x, module.linearbase.weight.data.clone(), module.biastensor.data.clone(), epstensor, *values )

        print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()

        print('tensorbiased linear custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, weight, biastensor, epstensor, *values  = ctx.saved_tensors
        print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['in_features','out_features']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)


        module= tensorbiased_linearlayer(paramsdict['in_features'], paramsdict['out_features'] , weight, biastensor )

        print('tesnorbased linaer custom input_.shape', input_.shape )
        eps=epstensor.item()
        X = input_.clone().detach().requires_grad_(True)
        R= lrp_backward(_input= X , layer = module , relevance_output = grad_output[0], eps0 = eps, eps=eps)

        print('linaer custom R', R.shape )
        #exit()
        return R,None,None




class posnegconv_tensorbiased(nn.Module):


    def _clone_module(self, module):
        clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                     **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)

    def __init__(self, tensorbiasedconv, ignorebias):
      super(posnegconv_tensorbiased, self).__init__()

      self.posconv=  tensorbiased_convlayer ( tensorbiasedconv.conv.weight, tensorbiasedconv.baseconv, tensorbiasedconv.inputfornewbias)
      self.negconv=  tensorbiased_convlayer ( tensorbiasedconv.conv.weight, tensorbiasedconv.baseconv, tensorbiasedconv.inputfornewbias)

      self.posconv.conv.weight= torch.nn.Parameter( tensorbiasedconv.conv.weight.data.clone().clamp(min=0) ).to(tensorbiasedconv.conv.weight.device)

      self.negconv.conv.weight= torch.nn.Parameter( tensorbiasedconv.conv.weight.data.clone().clamp(max=0) ).to(tensorbiasedconv.conv.weight.device)

      #self.posconv=self._clone_module(conv)
      #self.posconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(min=0) ).to(conv.weight.device)

      #self.negconv=self._clone_module(conv)
      #self.negconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(max=0) ).to(conv.weight.device)
    

      if True==ignorebias:
        self.posconv.inputfornewbias=None
        self.negconv.inputfornewbias=None
      else:
        self.posconv.biasmode='pos'
        self.negconv.biasmode='neg'

      print('posnegconv_tensorbiased done init')

    def forward(self,x):
        vp= self.posconv ( torch.clamp(x,min=0)  )
        vn= self.negconv ( torch.clamp(x,max=0)  )
        return vp+vn




class tensorbiasedconv2d_beta0_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, tensorbiasedclass, lrpignorebias):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(conv2dclass):
            #[module.in_channels, module.out_channels, module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']}

            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            values=[]
            for attr in propertynames:
              v = getattr(conv2dclass, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= conv2dclass.weight.device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= conv2dclass.weight.device)     
              else:
                print('v is neither int nor tuple. unexpected')
                exit()
              values.append(v)
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(tensorbiasedclass.baseconv)

        if tensorbiasedclass.baseconv.bias is None:
          bias=None
        else:
          bias= tensorbiasedclass.baseconv.bias.data.clone()
  
        if tensorbiasedclass.inputfornewbias is None:
          inputfornewbias= None
        else:
          inputfornewbias= tensorbiasedclass.inputfornewbias.data.clone()

        lrpignorebiastensor=torch.tensor([lrpignorebias], dtype=torch.bool, device= tensorbiasedclass.baseconv.weight.device)
        ctx.save_for_backward(x, tensorbiasedclass.baseconv.weight.data.clone(), bias, tensorbiasedclass.conv.weight.data.clone(), inputfornewbias,lrpignorebiastensor, *values ) # *values unpacks the list

        print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()

        print('tb2d custom forward')
        return tensorbiasedclass.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, conv2dweight, conv2dbias, newweight, inputfornewbias,lrpignorebiastensor, *values  = ctx.saved_tensors
        print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)

        if conv2dbias is None:
          conv2dclass=nn.Conv2d( **paramsdict, bias=False )
        else:
          conv2dclass=nn.Conv2d( **paramsdict, bias=True )
          conv2dclass.bias= torch.nn.Parameter(conv2dbias)
                
        #print('conv2dconstr')
        conv2dclass.weight= torch.nn.Parameter(conv2dweight)

        print('conv2dconstr weights')

        tensorbiasedclass=   tensorbiased_convlayer ( newweight = newweight , baseconv = conv2dclass , inputfornewbias = inputfornewbias )
        pnconv = posnegconv_tensorbiased(tensorbiasedclass, ignorebias= lrpignorebiastensor.item() )


        print('conv2d custom input_.shape', input_.shape )

        X = input_.clone().detach().requires_grad_(True)
        R= lrp_backward(_input= X , layer = pnconv , relevance_output = grad_output[0], eps0 = 1e-12, eps=0)

        print('conv2d custom R', R.shape )
        #exit()
        return R,None,None



#######################################################
#######################################################
# #base routines
#######################################################
#######################################################

def safe_divide(numerator, divisor, eps0,eps):
    return numerator / (divisor + eps0 * (divisor == 0).to(divisor) + eps * divisor.sign() )

'''
def lrp_backward(_input, layer, relevance_output, eps0, eps):
    """
    Performs the LRP backward pass, implemented as standard forward and backward passes.
    """
    relevance_output = relevance_output.clone().detach()
    with torch.enable_grad():
        Z = layer(_input)
        S = safe_divide(relevance_output, Z, eps0, eps)
        Z.backward(S)
        relevance_input = _input * _input.grad
    return relevance_input.data
'''


def lrp_backward(_input, layer, relevance_output, eps0, eps):
    """
    Performs the LRP backward pass, implemented as standard forward and backward passes.
    """
    relevance_output_data = relevance_output.clone().detach()
    with torch.enable_grad():
        Z = layer(_input)
    S = safe_divide(relevance_output_data, Z.clone().detach(), eps0, eps)

    #print('started backward')
    Z.backward(S)
    #print('finished backward')
    relevance_input = _input.data * _input.grad.data
    return relevance_input #.detach()


