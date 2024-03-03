import numpy as np
import deepxde as dde

from data_utils.data import DNO_Y_itransform, Theta_to_U, Theta_to_Z

def custom_mean_squared_error(y_true, y_pred):
    error = np.ravel((y_true - y_pred) ** 2)
    return np.mean(error)

class DeepXDE_DeepONet(object):

    def __init__(self,u_train,y_train,G_train,
                 u_test,y_test,G_test,
                 net,noise,N=2,lr=0.001,
                 model_dir="./",seed=3,model_str='',save_period=1000,epochs=1000,
                 nsteps=450,coarse=4,rank=2):
        
        self.N = N
        self.nsteps = nsteps
        self.coarse = coarse
        self.rank = rank
        self.noise= noise
        self.model_dir = model_dir
        self.seed = seed
        self.model_str = model_str
        self.epochs = epochs
        
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
        real_vals = DNO_Y_itransform(prediction_vals)
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
