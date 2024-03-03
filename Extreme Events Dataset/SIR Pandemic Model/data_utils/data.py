import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

def map_def(beta,gamma,delta,N,I0,T,dt,f):    
    S = np.zeros((int(T/dt),));
    S[0] = N;
    I= np.zeros((int(T/dt),));
    I[0] = I0;
    R = np.zeros((int(T/dt),));
    for tt in range(0,np.size(S)-1):
        # Ordinary different equations of the model
        dS = (-beta[tt]*I[tt]*S[tt] + delta*R[tt] - f[tt]) * dt;
        dI = (beta[tt]*I[tt]*S[tt] - gamma*I[tt] + f[tt]) * dt;
        dR = (gamma*I[tt] - delta*R[tt]) * dt;
        # Simple integration
        S[tt+1] = S[tt] + dS;
        I[tt+1] = I[tt] + dI;
        R[tt+1] = R[tt] + dR;
    return I

def Theta_to_U(noise,Theta,nsteps,coarse, udim):
        U1 = noise.get_sample(np.transpose(Theta))
        
        NN_grid = np.linspace(0,1,nsteps)
        Noise_grid = np.linspace(0,1,np.shape(U1)[0])
    
        U = np.zeros((np.shape(Theta)[0],nsteps))
        for i in range(0,np.shape(Theta)[0]):
            interp_func = InterpolatedUnivariateSpline(Noise_grid, U1[:,i], k=1)
            U[i,:] = interp_func(NN_grid)
        
        coarser_inds = np.linspace(0,nsteps-1,int(nsteps/coarse)).astype(int)
        U = U[:,coarser_inds]
        return U.astype(np.float32)

def Theta_to_Z(Theta,udim):
        if Theta.shape[1] == udim:
            Z = np.ones((Theta.shape[0], 1))
        else:
            Z = Theta[:,(udim+1):Theta.shape[1]]
        return Z.astype(np.float32)

# These functions are defined for normalizing, standardizing, or flatenining interal to DeepONet
def DNO_Y_transform(x):
    x_transform = np.log10(x)/10 - 0.5
    return x_transform.astype(np.float32)

def DNO_Y_itransform(x_transform):
        x = 10**((x_transform+0.5)*10)
        return x