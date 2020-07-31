# LRP_Pytorch_Resnets_Densenet
implements some common LRP rules to get explanations for 

Resnets and Densenet-121 as they come with the current pytorch model zoo, 

including batchnorm-Conv canonization and tensorbiased-conv2d layers coming up when canonizing densenets. 
uses custom backward via classes derived from torch.autograd.Function. should work on a GPU. See also innvestigate (Max Alber) with upcoming changes to tensorflow2.0 , lrp-toolbox (Sebastian), torchray (Ruth Fong), captum.

I wrote this code because I realized that there are hidden risks to get some things wrong, when implementing LRP, like:

--not fusing BN layers into convolutions or 

--not checking whether an LRP implementation will work with negative inputs, ( relu networks have them in the input only: image minus mean has neg values, for more complex approaches like feature map rescaling in some few shot methods those neg feature maps occur also in relu networks). 

Also to show that there are two different canonizations, depending on whether one has BN->conv as in resnets or BN->relu->conv as in densenets.

tested with >>> torch.__version__
'1.7.0.dev20200719'
Cuda 10.2 .

LRP-gamma is missing, i will add it after the AI.SG summer school because it is a coding task for them :) .


copy_resnet_wrap_poolnconv_v6.py  and copy_densenet_test6.py run the explanations. the \*onlycopy\*.py routines are only for validating the canonizations without the LRP wrappers inside.

Apologies for hordes of print statements like mushrooms after long rain, and for not writing much explanatory text. code has a structure. i will upload explanatory slides on that soon. I dont have the time for blog like texts, though i appreciate people who write deep explanations like jonathan hui on medium or lilian weng.
