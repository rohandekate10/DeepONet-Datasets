import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
import seaborn as sns

from data.data_functions import cur_DNO_Y_transform, cur_DNO_Y_itransform, Theta_to_U, Theta_to_Z

def custom_mean_squared_error(y_true, y_pred):
    error = np.ravel((y_true - y_pred) ** 2)
    return np.mean(error)

class DeepXDE_DeepONet(object):

    def __init__(self,Theta, n_wave_t, cur_Y, net, 
                 lr, epochs, N, model_dir, seed, save_period, model_str, 
                 coarse, udim, wDD,wVV,u_decimation_factor,n_init=3,make_plots=False):
        
        self.N = N
        self.nsteps = n_wave_t
        self.coarse = coarse
        self.rank = udim
        self.model_dir = model_dir
        self.seed = seed
        self.model_str = model_str
        self.epochs = epochs

        # Transform to U and G values
        u_train = Theta_to_U(alpha=Theta,nsteps=n_wave_t,coarse=coarse,udim=udim,n_wave_t=n_wave_t,wDD=wDD,wVV=wVV,u_decimation_factor=u_decimation_factor)
        y_train = Theta_to_Z(Theta,udim)
        G_train = cur_DNO_Y_transform(cur_Y)
        
        
        if make_plots == True:
            print(f"Shape of u_train:{u_train.shape}")
            print(f"Shape of y_train:{y_train.shape}")
            print(f"Shape of G_train:{G_train.shape}")
            # Make Plots
            n_datapoints = u_train.shape[1]
            separation_idx = n_datapoints
            fig,(ax1,ax2) = plt.subplots(2,1)
            for row in range(n_init):
                ax1.plot(np.concatenate((u_train[row], G_train[row]), axis=0), "-." ,label = f"Training Trajectory {row}")
                ax2.plot(G_train[row], "-o", label = f"G_train {row}")
            ax1.axvline(x=separation_idx, color='r', linestyle='--', label="<- u_train | G_train ->")
            ax1.grid()
            ax1.set_ylabel(f"VBM $(Nm)$")
            ax1.legend()
            ax2.grid()
            ax2.set_ylabel(f"VBM $(Nm)$")
            ax2.legend()
            plt.tight_layout()
            sns.despine(trim=True)
            plt.show()

        if make_plots == False:
            # Making Dummy Variable for testing since it is redundant right now
            u_test = u_train[0,:].reshape((1,np.size(u_train[0,:]))).astype(np.float32)
            y_test = y_train[0,:].reshape((1,np.size(y_train[0,:]))).astype(np.float32)
            G_test = G_train[0,:].reshape((1,np.size(G_train[0,:]))).astype(np.float32)
            
            self.data = dde.data.Triple(X_train=(u_train,y_train), y_train=G_train, X_test=(u_test,y_test), y_test=G_test)
            # Initilize a list of DeepONet models
            self.modelN = list()
            for i in range(0,N):
                self.modelN = np.append(self.modelN,dde.Model(self.data,net))
                # Compile model N
                self.modelN[i].compile("adam", lr=lr, metrics=[custom_mean_squared_error])
                checker = dde.callbacks.ModelCheckpoint(self.model_dir+"model/N"+str(i)+"seed"+str(self.seed)+"_"+self.model_str+"_model.ckpt", save_better_only=False, period=save_period)
                self.modelN[i].losshistory, self.modelN[i].train_state = self.modelN[i].train(epochs=epochs, callbacks=[checker]) #Training Model batch_size = 10000
            # Append all models together
            self.model = list()
            for i in range(0,N):
                self.model = np.append(self.model,self.modelN[i])

    def predict(self,Thetanew):
        prediction_vals = self.predict_all(Thetanew)
        real_vals = cur_DNO_Y_itransform(prediction_vals)
        mean_vals = np.mean(real_vals,axis = 1).reshape(np.shape(Thetanew)[0],1)
        var_vals = np.var(real_vals,axis = 1).reshape(np.shape(Thetanew)[0],1)
        
        return mean_vals, var_vals
    
    def predict_all(self,Thetanew):
        # This computes and reports all ensemble values
        prediction_vals = np.zeros((np.shape(Thetanew)[0], self.N))
        # noise,Theta,nsteps,coarse
        U = Theta_to_U(self.noise,Thetanew, self.nsteps, self.coarse, self.rank) # This step can be memory intensive.
        Z = Theta_to_Z(Thetanew, self.rank)
        UZnew = (U,Z)
        for i in range(0,self.N):
            self.model[i].restore(self.model_dir+"model/N"+str(i)+"seed"+str(self.seed)+"_"+self.model_str+"_model.ckpt-" + str(self.epochs) + ".pt", verbose=0)
            temp = self.model[i].predict((UZnew))
            prediction_vals[:,i] = temp.reshape(np.shape(UZnew[0])[0],)
        return prediction_vals
