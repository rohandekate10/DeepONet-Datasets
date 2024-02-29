import os
import numpy as np
import sklearn as sk


#####################
# IO Helper Functions
#####################

def make_dirs(output_path, err_save_path, model_dir, as_dir, fig_save_path, intermediate_data_dir):
    #
    # Why oh why can't mkdir() fail silently if a direktory already exists?
    #
    
    try:
        os.mkdir(output_path)
    except OSError as error:
        print(error)
    try:
        os.mkdir(err_save_path)
    except OSError as error:
        print(error)
    try:
        os.mkdir(model_dir)
    except OSError as error:
        print(error)    
    try:
        os.mkdir('{}/model'.format(model_dir)) # 'cause DeepONet has some built in pathing
    except OSError as error:
        print(error)
    try:
        os.mkdir(as_dir)
    except OSError as error:
        print(error)
    try:
        os.mkdir(fig_save_path)
    except OSError as error:
        print(error)
    try:
        os.mkdir(intermediate_data_dir)
    except OSError as error:
        print(error)

def load_wave_data(data_path, model_suffix):
    #
    # Load some precomputed LAMP data
    #
       
    wave_TT_filename = '{}TT{}.txt'.format(data_path, model_suffix)
    wave_DD_filename = '{}DD{}.txt'.format(data_path, model_suffix)
    wave_VV_filename = '{}VV{}.txt'.format(data_path, model_suffix)
       

    wTT = np.loadtxt(wave_TT_filename)
    wDD = np.loadtxt(wave_DD_filename)
    wVV = np.loadtxt(wave_VV_filename)
       
    return wTT, wDD, wVV

def load_vbm_lhs_data(data_path, model_suffix, trim=True):
    vbm_TT_lhs_filename = '{}kl-2d{}-tt.txt'.format(data_path, model_suffix)
    vbm_zz_lhs_filename = '{}kl-2d{}-vbmg.txt'.format(data_path, model_suffix)
    vbm_aa_lhs_filename = '{}kl-2d{}-design.txt'.format(data_path, model_suffix)
    
    vTTlhs = np.loadtxt(vbm_TT_lhs_filename)
    vZZlhs = np.loadtxt(vbm_zz_lhs_filename)
    vAAlhs = np.loadtxt(vbm_aa_lhs_filename)
    
    if trim :
        vZZlhs = vZZlhs[0:625, :]   # minor accounting error during LAMP problem design
        vAAlhs = vAAlhs[0:625, :]        
    
    return vTTlhs, vZZlhs, vAAlhs
        
def load_vbm_mc_data(data_path, model_suffix):
    vbm_TT_mc_filename = '{}kl-2d{}-test-tt.txt'.format(data_path, model_suffix)
    vbm_zz_mc_filename = '{}kl-2d{}-test-vbmg.txt'.format(data_path, model_suffix)
    vbm_aa_mc_filename = '{}kl-2d{}-test-design.txt'.format(data_path, model_suffix)
    
    vTTmc = np.loadtxt(vbm_TT_mc_filename)
    vZZmc = np.loadtxt(vbm_zz_mc_filename)
    vAAmc = np.loadtxt(vbm_aa_mc_filename)
    
    return vTTmc, vZZmc, vAAmc

def load_gpr_precomputed(gpr_pdf_path, ndim):
    if (ndim > 6) :
        qdim = 6
    else :
        qdim = ndim
    
    qq_xx_filename = '{}{}-40-modes-bins.txt'.format(gpr_pdf_path, qdim)
    qq_pp_filename = '{}{}-40-modes-hist.txt'.format(gpr_pdf_path, qdim)
    
    if ndim >= 6 :
        # b/c the GPR VBM pdf is real bad for 6D, skip straight to the true MC
        # data, equivalent to \inf D
        mm_xx_filename = '{}mc-vbm-bins.txt'.format(gpr_pdf_path)
        mm_pp_filename = '{}mc-vbm-hist.txt'.format(gpr_pdf_path)
    else :
        mm_xx_filename = '{}{}-40-vbm-bins.txt'.format(gpr_pdf_path, ndim)
        mm_pp_filename = '{}{}-40-vbm-hist.txt'.format(gpr_pdf_path, ndim)
    
    qq_xx = np.loadtxt(qq_xx_filename)
    qq_xx = 1/2*(qq_xx[0:-1] + qq_xx[1::])  # b/c big dumb I saved it wrong
    qq_pp = np.loadtxt(qq_pp_filename)
    mm_xx = np.loadtxt(mm_xx_filename)
    mm_pp = np.loadtxt(mm_pp_filename)
    
    return qq_xx, qq_pp, mm_xx, mm_pp

############
# PCA Stuff
############

def project_onto_vector(x, v):
     a = np.dot(x, v) / np.dot(v, v)
     return a
 
    #
    # PCA transform of VBM!
    #
    # sklearn doesn't automatically normalize the PCA components, so we do that
    # by hand
    #
    # Actually, sklearn doesn't do PCA the same way I've been doing it, so I should
    # use my other method.  Probably smart PCA has various regularization stuff
    # for statisticians that I don't want
    #

def pca_transform_z_2_q(vZZlhs, vZZmc, sklearn_pca_algo = False, n_q_modes=6) :
    
    n_lhs_data = vZZlhs.shape[0]
    
    if sklearn_pca_algo :
        q_pca = sk.decomposition.PCA(n_components = n_q_modes)
        q_pca.fit(vZZmc)
        
        #print(q_pca.explained_variance_ratio_)
        #print(q_pca.singular_values_)
        
        q_lambda_mat = q_pca.get_covariance()
        q_lambda_list = np.zeros([n_q_modes,])
        for k in range(0, n_q_modes):
            q_lambda_list[k] = q_lambda_mat[k ,k]
        
        QQ_raw = q_pca.transform(vZZlhs)
        QQ = np.zeros(QQ_raw.shape)
        
        for k in range(0, n_q_modes):
            QQ[:, k] = QQ_raw[:, k] / np.sqrt( q_lambda_list[k])
        
    else : 
        vv_var = np.var(vZZmc.ravel())
        vv_norm = vZZmc/np.sqrt(vv_var)
        
        CC = np.matmul(np.transpose(vv_norm), vv_norm)
        CC = CC/n_lhs_data
        w_vbm, v_vbm = np.linalg.eig(CC)
        
        QQ = np.zeros([n_lhs_data, n_q_modes])
        vZZlhs_norm = vZZlhs/np.sqrt(vv_var)
        
        for k in range(0, n_q_modes):
             aa = project_onto_vector(vZZlhs_norm, v_vbm[:, k])            
             QQ[:, k] = aa/np.sqrt(w_vbm[k])
             
    return QQ, w_vbm, v_vbm, vv_var

#######################
# DNO Transform Things
#######################

    #    
    # These functions are defined for normalizing, standardizing, or flatenining interal to DeepONet
    #
    # Ethan sez:  decimation_factor = 2 is good, but might even be too low
    # 

def DNO_Y_transform(x, decimation_factor = 3):
    x_transform = x/decimation_factor
    return x_transform

def DNO_Y_itransform(x_transform, decimation_factor = 3):
    x = x_transform*decimation_factor
    return x