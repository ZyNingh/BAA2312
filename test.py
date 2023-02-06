from matplotlib import pyplot as PLT
from bilevelmri.experiment_setup import learn, compute_statistics
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
from scipy import fftpack

path = "//home//leidenschaftchen//MRI//BAA2312//image//sub-093-anat-sub-093_run-01_T1w.nii.gz"
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

params = {
    'model': {
        'n1': n1,
        'n2': n2
    },
    'alg_params': {
        'll_sol': {
            'maxit': 10,
            'tol': 1e-10
        },
        'lin_sys': {
            'maxit': 10,
            'tol': 1e-6
        },
        'LBFGSB': {
            'maxit': 10,
            'pgtol': 1e-8
        }
    }
}

A = Grad()
reg_func = Smoothed1Norm(gamma=1e-2)


def penalty(p):
    return l1_disc_penalty(p[:-2], beta=(.1, .1))

# tune alpha on full sampling pattern to get initialisation
tuned_alpha = learn(data, 1e-3, [(0, np.inf)], alpha_parametrisation, A,
                    reg_func, least_squares, lambda p: torch.zeros_like(p),
                    params)

p_init = np.ones(n1 * n2 + 2)
p_init[-1] = 1e-2
p_init[-2] = tuned_alpha['p']
p_bounds = [(0., 1.) for _ in range(n1 * n2)]
p_bounds.append((0, np.inf))
p_bounds.append((1e-2, 1e-2))
# learn sampling pattern
result = learn(data, p_init, p_bounds, free_parametrisation, A, reg_func,
               least_squares, penalty, params)

stats = compute_statistics(data, result['p'], A, reg_func, free_parametrisation, params)

imageio.imwrite("SDFJ1.png",torch.sqrt(torch.sum(data['x'][0, :, :, :]**2, dim=2)))

imageio.imwrite("SDFJ12.png",fftpack.fftshift(result['p'][:-2].reshape(n1, n2)))


imageio.imwrite("SDFJ123.png",torch.sqrt(torch.sum(stats['recons'][0, :, :, :]**2, dim=2)))

