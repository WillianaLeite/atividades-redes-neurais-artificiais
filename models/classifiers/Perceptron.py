import pandas as pd
import numpy as np
import random
from models import plots

class Perceptron_Simple():
    
    def __init__(self, n_epochs, learning_rate=0.1):
        
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        
    
    def __calc_u(self, vect_input, vect_weigths):
        return sum([ w*x for w, x in zip(vect_input, vect_weigths) ])
    
    def __func_ativacao_degrau(self, u):
        if u > 0: return 1
        else: return 0
        
    def __new_weigths(self, weigth_old, taxa_aprendizado, error, x):
        return weigth_old + taxa_aprendizado * error * x
    
    def run_epoch(self, X, y, taxa_aprendizado, vect_weigths, verbose, verbose_debug):

        df_train = X.copy()
        df_train['target'] = y

        list_error = []
        for index, row in df_train.iterrows():

            vect_input = df_train.drop(['target'], axis=1).iloc[index, :].tolist()
            #vect_input = vect_input.insert(0, -1)
            
            d = df_train.iloc[index, -1] # Desejável 

            u = self.__calc_u(vect_input, vect_weigths) 
            y = self.__func_ativacao_degrau(u) # Predito
            
            error = d - y
            
            if verbose_debug: 
                print(f'\n\n\nLinha {index}: \nInput: {vect_input}\n\nVect_weigths: {vect_weigths}\nU: {u}\nY: {y}\nError: {error}')
            
            list_error.append(error)

            vect_weigths = [self.__new_weigths(weigth_old, taxa_aprendizado, error, x) 
                            
                            for x, weigth_old in zip(vect_input, vect_weigths)] 

        return vect_weigths, list_error
    
    def fit(self, X, y, verbose=False, verbose_debug=False, plot_result_epoch=False):
        
        #self.vect_weihts = [random.uniform(-1, 1) for i in range(len(X.columns))]
        self.vect_weihts = [random.random() for i in range(len(X.columns))] # Positivo
        self.vect_weihts.insert(0, -1) # Peso Negativo
        
        X.insert(0, 'x0', -1)
        
        df = X.reset_index(drop=True).copy()
        df['targuet'] = y
        
        list_sum_error = []
        
        for epoch in range(self.n_epochs):

            df = df.sample(frac=1).reset_index(drop=True) # O sample faz o shuffle e pegar uma frac 100%
            df = df.reset_index(drop=True)
            
            X, y = df.drop(['targuet'], axis=1), df['targuet']
            
            self.vect_weihts, list_error = self.run_epoch(X, y, self.learning_rate, self.vect_weihts, verbose,verbose_debug)
            
            mean_error = sum(list_error) / len(list_error)
            
            if verbose or verbose_debug:     
                print('\n\n')
                print('*' *125)
                print(f'* Epoch: {epoch}',  ' '*112, '*')
                print('*' *125)
                print(f'Error médio: {mean_error}')
                
            list_sum_error.append((epoch, sum(list_error)))
            
            if [0.0]* len(list_error) == list_error: # A lista de todos os erros da época tá zerada?
                
                break
        
        if plot_result_epoch:
            plots.plot_error_epoch(list_sum_error)

    def predict(self, X):
        
        X.insert(0, 'x0', -1)
        
        df_test = X.reset_index(drop=True).copy()
        
        list_predict = []
        
        for index, row in df_test.iterrows():
            
            vect_input = df_test.iloc[index, :].tolist()
            
            u = self.__calc_u(vect_input, self.vect_weihts) 
            y = self.__func_ativacao_degrau(u)
            
            list_predict.append(y)

        return list_predict