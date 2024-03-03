import numpy as np

# Create the Theta_u to U map for saving localling to be called by Matlab MMT files
def Save_U(Theta,noise,nsteps,rank):
    y0 = np.zeros((1,nsteps),dtype=np.complex_)
    xr = Theta[0:rank]
    xi = Theta[rank:(2*rank)]
    x = xr + 1j*xi
    y0 = noise.get_sample(x)
    return y0

# Mapping Theta to U with only one input
def map_def(theta_rank):
    rank = int(theta_rank[1])
    theta = theta_rank[0].reshape(rank*2,)
    nsteps = 512
    y0 = Save_U(theta,nsteps,rank)
    return y0

# Create the X to U map, which is actually theta to U
def Theta_to_U(noise,Theta,nsteps,coarse,rank):
    # We can also coarsen the steps 512 is likely extra fine for Deeponet
    Theta = np.atleast_2d(Theta)
    U = np.zeros((np.shape(Theta)[0],2*int(nsteps/coarse)),dtype=np.complex_)

    # Determine real and imaginary inds
    dim = int(np.shape(Theta)[1]/2)
    xr = Theta[:,0:(dim)]
    xi = Theta[:,dim:dim*2]
    x = xr + 1j*xi
    Us = np.transpose(noise.get_sample(x))
    coarser_inds = np.linspace(0,nsteps-1,int(nsteps/coarse)).astype(int)

    real_inds = np.linspace(0,nsteps/coarse*2-2,int(nsteps/coarse)).astype(int)
    imag_inds = np.linspace(1,nsteps/coarse*2-1,int(nsteps/coarse)).astype(int)
    
    U[:,real_inds] = np.real(Us[:,coarser_inds])
    U[:,imag_inds] = np.imag(Us[:,coarser_inds])
    return U

def Theta_to_Z(Theta,rank):
    Z = np.ones((Theta.shape[0], 1))
    return Z


# These functions are defined for normalizing, standardizing, or flatenining interal to DeepONet
def DNO_Y_transform(x):
    x_transform = x
    return x_transform

def DNO_Y_itransform(x_transform):
    x = x_transform
    return x