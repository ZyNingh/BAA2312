import pydicom
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import skimage.io as io
import imageio
from scipy import fftpack
import torch
from pytorch_wavelets import DTCWTForward, DTCWTInverse, DWTForward
torch.set_default_dtype(torch.float64)

for i in range(7):
    patt = 'J2'+str(i+1)+'.png'

    im = imageio.imread(patt)


        #plt.imshow(img.pixel_array)
        #plt.show()
        #print(img.pixel_array.shape)

    im = im.astype(np.float64)

    MavVl = float(np.amax(im))
    im = im / MavVl

    t = torch.from_numpy(im)
    a = torch.zeros([1,1,256,256])
    a[0,0] = t
    newimg = torch.zeros([1,1,256,256])
    dwt = DWTForward(J=3, wave='db4', mode='per')

    rel, rell = dwt(a)
    Wid = rel.shape[2]
    Hig = rel.shape[3]
    newimg[0,0,0:Wid,0:Hig] = rel
    newimg[0,0,Wid:Wid*2,0:Hig] = rell[2][0,0,0]
    newimg[0,0,00:Wid,Hig:2*Hig] = rell[2][0,0,1]
    newimg[0,0,Wid:2*Wid,Hig:2*Hig] = rell[2][0,0,2]
    newimg[0,0,Wid*2:Wid*4,0:2*Hig] = rell[1][0,0,0]
    newimg[0,0,00:Wid*2,2*Hig:4*Hig] = rell[1][0,0,1]
    newimg[0,0,Wid*2:Wid*4,2*Hig:4*Hig] = rell[1][0,0,2]
    newimg[0,0,00:Wid*4,4*Hig:] = rell[0][0,0,0]
    newimg[0,0,Wid*4:,0:4*Hig] = rell[0][0,0,1]
    newimg[0,0,Wid*4:320,4*Hig:] = rell[0][0,0,2]
    imageio.imsave('wa'+patt, newimg[0,0])
    y = torch.cat(tuple(newimg[0,0,i] for i in range(256)),0)
    x = np.arange(y.shape[0])
    plt.subplot(7, 1, i+1)
    plt.plot(x,y)
plt.savefig("resultbyrow")