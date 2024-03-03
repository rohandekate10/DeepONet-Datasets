import numpy as np
import utils.lamp_helper_functions as hf

#
# Convert from wave episode coefficient form to wave episode time series
#
def Theta_to_U(alpha, nsteps, coarse, udim, n_wave_t,wDD,wVV,u_decimation_factor): 
    
    
    if alpha.shape[0] == 0 :
        U1 = np.zeros([n_wave_t, 1])
        for k in range(0, udim):
            U1 = U1 + alpha[k]*np.sqrt(wDD[k])*wVV[:, k]
    else :
        U1 = np.zeros([alpha.shape[0], n_wave_t])
        for k in range(0, udim):
            for j in range(0, alpha.shape[0]):
                U1[j, :] = U1[j, :] + alpha[j, k]*np.sqrt(wDD[k])*wVV[:, k]
                
    coarser_inds = np.linspace(0,nsteps-1,int(nsteps/coarse)).astype(int)
    U = U1[:,coarser_inds].astype(np.float32)
    return U/u_decimation_factor

#
# Dummy function, map everything to 1
#

def Theta_to_Z(Theta,udim=10):
    if Theta.shape[1] == udim:
        Z = np.ones((Theta.shape[0], 1))
    else:
        Z = Theta[:,(udim+1):Theta.shape[1]]
    return Z.astype(np.float32)

#
# Lambda the DNO transforms, to include the decimation faktor parameter
#

cur_DNO_Y_transform = lambda x,y_decimation_factor=2 : hf.DNO_Y_transform(x, decimation_factor=y_decimation_factor)
cur_DNO_Y_itransform = lambda x,y_decimation_factor=2 : hf.DNO_Y_itransform(x, decimation_factor=y_decimation_factor)