from matplotlib import pyplot as PLT
from bilevelmri.linear_ops.gradients import Grad
from bilevelmri.functionals import Smoothed1Norm
from bilevelmri.loss_functions import least_squares
from bilevelmri.penalty_functions import l1_disc_penalty
from bilevelmri.parametrisations import alpha_parametrisation, free_parametrisation
import torch
import numpy as np
torch.set_default_dtype(torch.float64)
import SimpleITK as sitk
import skimage.io as io
import imageio

path= "//home//leidenschaftchen//MRI//BAA2312//image//sub-093-anat-sub-093_run-01_T1w.nii.gz"
PREimg = sitk.ReadImage(path)
PREdata = sitk.GetArrayFromImage(PREimg)


rawImage = PREdata[50]
MavVl = float(np.amax(rawImage))
rawImage = rawImage / MavVl

image = torch.tensor(rawImage)

#image = torch.tensor(PREdata[115])
x = torch.zeros(1,256,256,2)
x[0,:,:,0] = image
#for i in range(10):
#    x[i,:,:,0] = torch.tensor(PREdata[35+i])


y = torch.fft(x, signal_ndim=2, normalized=True) + 0.03 * torch.randn_like(x)
data = {'x': x, 'y': y}
n1, n2 = x.shape[1:3]

imageio.imwrite("raw imame.jpg", x[0,:,:,0])

imageio.imwrite("rawksapce.jpg", y[0,:,:,0], cmap = "gray")

