
import os

import PIL.Image

from getimagenetclasses import *
from torch.utils.data import Dataset


###################################################
##################################################3
###################################################33

#preprocessing: https://pytorch.org/docs/master/torchvision/models.html
#transforms: https://pytorch.org/docs/master/torchvision/transforms.html
#grey images, best dealt before transform
# at first just smaller side to 224, then 224 random crop or centercrop(224)
#can do transforms yourself: PIL -> numpy -> your work -> PIL -> ToTensor()

class dataset_imagenetvalpart_nolabels(Dataset):
  def __init__(self, root_dir, maxnum, skip=0, transform=None):
  #def __init__(self, root_dir, xmllabeldir, synsetfile, maxnum, skip=0, transform=None):

    self.root_dir = root_dir
    #self.xmllabeldir=xmllabeldir
    self.transform = transform
    self.imgfilenames=[]
    self.labels=[]
    self.ending=".JPEG"

    self.clsdict=get_classes()


    #indicestosynsets,self.synsetstoindices,synsetstoclassdescr=parsesynsetwords(synsetfile)

    
    allfiles=[]
    for root, dirs, files in os.walk(self.root_dir):
      allfiles.extend(files)
    allfiles = sorted(allfiles)


    for ct,name in enumerate(allfiles):
      if ct < skip:
        continue
      

      nm=os.path.join(root, name)
      print(nm)
      if (maxnum >0) and ct>= (maxnum + skip):
        break
      self.imgfilenames.append(nm)
      #label,firstname=parseclasslabel(self.filenametoxml(nm) ,self.synsetstoindices)
      #self.labels.append(label)

    #print(self.imgfilenames)
    #exit()

   
  def filenametoxml(self,fn):
    f=os.path.basename(fn)
     
    if not f.endswith(self.ending):
      print('not f.endswith(self.ending)')
      exit()
       
    f=f[:-len(self.ending)]+'.xml'
    f=os.path.join(self.xmllabeldir,f) 
     
    return f


  def __len__(self):
      return len(self.imgfilenames)

  def __getitem__(self, idx):
    image = PIL.Image.open(self.imgfilenames[idx]).convert('RGB')

    label= -1 #self.labels[idx]

    if self.transform:
      image = self.transform(image)

    #print(image.size())

    sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx]}

    return sample
