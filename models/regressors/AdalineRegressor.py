import pandas as pd
import numpy as np
import random
from models import plots

class AdalineRegressor():
    
    def __init__(self, n_epochs, learning_rate=0.1):
        
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        
    
    def __calc_u(self, vect_input, vect_weigths):
        return sum([ w*x for w, x in zip(vect_input, vect_weigths) ])
    
    def __new_weigths(self, weigth_old, taxa_aprendizado, error, x):
        return weigth_old + taxa_aprendizado * error * x
    
    def run_epoch(self, X, y, taxa_aprendizado, vect_weigths, verbose, verbose_debug):

        df_train = X.copy()
        df_train['target'] = y

        list_error = []
        for index, row in df_train.iterrows():

            vect_input = df_train.drop(['target'], axis=1).loc[index, :].tolist()
            if verbose_debug: 
                print('\n\n\n1:', df_train.drop(['target'], axis=1).loc[index, :].tolist())
                print('2:', df_train.loc[index, :].tolist())
                print('3:', vect_input)
            d = df_train.iloc[index, -1]  

            u = self.__calc_u(vect_input.copy(), vect_weigths.copy()) 
            
            y = u
            
            error = (d-u) ** 2 
                        
            if verbose_debug: 
                print(f'Linha {index}: \nInput: {vect_input}\nVect_weigths: {vect_weigths}\nd: {d}\nY: {y}\nError: {error}')
            
            list_error.append(error)

            vect_weigths = [self.__new_weigths(weigth_old, taxa_aprendizado, error, x) 
                            
                            for x, weigth_old in zip(vect_input.copy(), vect_weigths.copy())] 
        
        mse = sum(list_error) / len(list_error)
        
        if verbose_debug: 
            print(f'\nMSE: {mse}')
        
        return vect_weigths, mse
    
    def fit(self, X, y, verbose=False, verbose_debug=False, plot_result_epoch=False):
        
        self.vect_weihts = [random.random() for i in range(len(X.columns))]
        self.vect_weihts.insert(0, -1) # w0
        
        X.insert(0, 'x0', 1)
        
        df = X.reset_index(drop=True).copy()
        df['targuet'] = y
        
        list_mse = []
        
        list_sum_error = []
        for epoch in range(self.n_epochs):

            df = df.sample(frac=1).reset_index(drop=True) # O sample faz o shuffle e pega uma frac 100%
            df = df.reset_index(drop=True)
            
            X, y = df.drop(['targuet'], axis=1).copy(), df['targuet']
            
            self.vect_weihts, mse = self.run_epoch(X.copy(), y, self.learning_rate, self.vect_weihts, verbose,verbose_debug)
            
            list_mse.append(mse)
            
            if verbose or verbose_debug:     
                print('\n\n')
                print('*' *125)
                print(f'* Epoch: {epoch}',  ' '*112, '*')
                print('*' *125)
                print(f'MSE: {mse}')
                
            if float(mse) == 0.0:
                break
        
        if plot_result_epoch:
            plots.plot_error_epoch(zip(range(self.n_epochs), list_mse))

    def predict(self, X):
        
        X.insert(0, 'x0', 1)
        
        df_test = X.reset_index(drop=True).copy()
        
        list_predict = []
        
        for index, row in df_test.iterrows():
            
            vect_input = df_test.iloc[index, :].tolist()
            
            u = self.__calc_u(vect_input, self.vect_weihts) 
            y = u
            
            list_predict.append(y)

        return list_predict