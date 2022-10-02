import os
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import seaborn as sns


class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X):

        #Preprocessing 
        window = 200;
        telescope = 100;

        X = X.numpy()
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)

        future = X[-window:]
        future = (future - X_min) / (X_max - X_min)
        future = np.expand_dims(future, axis=0)

        #Predict
        reg_telescope = 900
        reg_predictions = np.array([])
        X_temp = future

        #range() takes as input (start, stop , step)
        for reg in range(0,reg_telescope,telescope):
            pred_temp = self.model.predict(X_temp)      #after an iteration pred_temp has shape:  200, 100, 7  ; at second iteration shape: 200, 100, 7
            if(len(reg_predictions)==0):                #only at first iteration we enter the if clause shape:  200,100,7
                reg_predictions = pred_temp
            else:
                reg_predictions = np.concatenate((reg_predictions,pred_temp),axis=1)    #we enter at all the iterations after the first one, at second iteration shape:200,200,7; at third iteration shape: 200,300,7 
            X_temp = np.concatenate((X_temp[:,telescope:,:],pred_temp), axis=1)     #after first iteration X_temp ha shape: 200, 100, 7  (the first and last dimensions don't get concatenated since we are concatenating over the axis = 1)  at second iteration shape: 200,100,7

        reg_predictions = reg_predictions[:,:864,:]

        #Postprocessing
        out = np.reshape(reg_predictions, (864, 7))
        out = out * (X_max - X_min) + X_min
        out = np.reshape(out, (864, 7))
        out = tf.convert_to_tensor(out)

        return out
