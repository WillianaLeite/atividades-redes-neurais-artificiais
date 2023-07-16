import pandas as pd
import numpy as np
import random
from models import plots
import math

class Perceptron_Generic():
    
    def __init__(self, n_epochs, learning_rate=0.1, func_ativacao_personalizada=None):
        
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.name_func_ativacao = func_ativacao_personalizada
        
        if func_ativacao_personalizada == 'degrau':
            self.func_ativacao = self.__degrau
        
        elif func_ativacao_personalizada == 'logistica':
            self.func_ativacao = self.__sigmoide_logistica
        
        elif func_ativacao_personalizada == 'tangente':
            self.func_ativacao = self.__sigmoide_tangente
        
        else:
            self.func_ativacao = func_ativacao_personalizada
        
    
    def __calc_u(self, vect_input, vect_weigths):
        return sum([ w*x for w, x in zip(vect_input, vect_weigths) ])
    
    def __degrau(self, u, threshold=0):
        if u >= threshold: return 1
        else: return 0
        
    def __sigmoide_logistica(self, u):
        return 1 / (1 + math.exp(-u))
    
    def __sigmoide_tangente(self, u):
        return (1 - math.exp(-u)) / (1 + math.exp(-u))
        
    def __new_weigths(self, weigth_old, taxa_aprendizado, error, x):
        return weigth_old + taxa_aprendizado * error * x
    
    def run_epoch(self, X, y, taxa_aprendizado, vect_weigths, verbose, verbose_debug):

        df_train = X.copy()
        df_train['target'] = y

        list_error = []
        for index, row in df_train.iterrows():

            vect_input = df_train.drop(['target'], axis=1).iloc[index, :].tolist()
            
            d = df_train.iloc[index, -1] # DesejÃ¡vel 

            u = self.__calc_u(vect_input, vect_weigths) 
            y = self.func_ativacao(u) # Predito
            
            error = d - y
            
            if verbose_debug: 
                print(f'\n\n\nLinha {index}: \nInput: {vect_input}\n\nVect_weigths: {vect_weigths}\nU: {u}\nY: {y}\nD: {d}\nError: {error}')
            
            list_error.append(error)

            vect_weigths = [self.__new_weigths(weigth_old, taxa_aprendizado, error, x) 
                            
                            for x, weigth_old in zip(vect_input, vect_weigths)] 

        return vect_weigths, list_error
        
    
    def fit(self, X, y, verbose=False, verbose_debug=False, plot_result_epoch=False, vect_weihts_start=None):
        
        if vect_weihts_start is None:
            self.vect_weihts = [random.random() for i in range(len(X.columns))]
            self.vect_weihts.insert(0, -1) # w0
            X.insert(0, 'x0', -1)
        
        else:
            self.vect_weihts = vect_weihts_start
        
        df = X.reset_index(drop=True).copy()
        df['targuet'] = y
        if self.name_func_ativacao == 'tangente':
            df['targuet'] = df['targuet'].apply(lambda y: y if y == 1 else -1)
        
        list_sum_error = []
        
        for epoch in range(self.n_epochs):

            df = df.sample(frac=1).reset_index(drop=True) # O sample faz o shuffle e pega uma frac 100%
            df = df.reset_index(drop=True)
            
            X, y = df.drop(['targuet'], axis=1), df['targuet']
            
            self.vect_weihts, list_error = self.run_epoch(X, y, self.learning_rate, self.vect_weihts, verbose,verbose_debug)
            
            mean_error = sum(list_error) / len(list_error)
            
            if verbose or verbose_debug:     
                print('\n\n')
                print('*' *125)
                print(f'* Epoch: {epoch}',  ' '*112, '*')
                print('*' *125)
                print(f'Error mÃ©dio: {mean_error}')
                
            list_sum_error.append((epoch, sum(list_error)))
            
            if [0.0]* len(list_error) == list_error:
                
                break
        
        if plot_result_epoch:
            plots.plot_error_epoch(list_sum_error)
            
        return self

    def predict(self, X):
        
        if 'x0' not in X.columns:
            X.insert(0, 'x0', -1)
        
        df_test = X.reset_index(drop=True).copy()
        
        list_predict = []
        
        if len(df_test) > 1:
        
            for index, row in df_test.iterrows():

                vect_input = df_test.iloc[index, :].tolist()

                u = self.__calc_u(vect_input, self.vect_weihts) 
                y = self.func_ativacao(u)

                if self.name_func_ativacao == 'tangente':
                    y = self.__degrau(y, threshold=0.00)

                elif self.name_func_ativacao == 'logistica':
                    y = self.__degrau(y, threshold=0.50)

                list_predict.append(y)

            return list_predict
        
        elif len(df_test) == 1:
            
            vect_input = df_test.iloc[0, :].tolist()

            u = self.__calc_u(vect_input, self.vect_weihts) 
            y = self.func_ativacao(u)

            if self.name_func_ativacao == 'tangente':
                y = self.__degrau(y, threshold=0.00)

            elif self.name_func_ativacao == 'logistica':
                y = self.__degrau(y, threshold=0.50)
                
            return y, u