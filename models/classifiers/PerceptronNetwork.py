import pandas as pd
import numpy as np
import random
from models import plots, utils

class Neuron():
    
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
            
            d = df_train.iloc[index, -1] # DesejÃƒÂ¡vel 

            u = self.__calc_u(vect_input, vect_weigths) 
            y = self.__func_ativacao_degrau(u) # Predito
            
            error = d - y
            
            if verbose_debug: 
                print(f'\n\n\nLinha {index}: \nInput: {vect_input}\n\nVect_weigths: {vect_weigths}\nU: {u}\nY: {y}\nError: {error}')
            
            list_error.append(error)

            vect_weigths = [self.__new_weigths(weigth_old, taxa_aprendizado, error, x) 
                            
                            for x, weigth_old in zip(vect_input, vect_weigths)] 

        return vect_weigths, list_error
    
    def fit(self, X, y, vect_weihts_start, verbose=False, verbose_debug=False, plot_result_epoch=False):
        
        self.vect_weihts = vect_weihts_start
        
        df = X.reset_index(drop=True).copy()
        df['targuet'] = y
        
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
                print(f'Error medio: {mean_error}')
                
            list_sum_error.append((epoch, sum(list_error)))
            
            if [0.0]* len(list_error) == list_error:
                
                break
        
        if plot_result_epoch:
            plots.plot_error_epoch(list_sum_error)
            
        return self

    def predict(self, vect_input):
        
        u = self.__calc_u(vect_input, self.vect_weihts) 
        y = self.__func_ativacao_degrau(u)

        return y, u
    
        


class Perceptron_Network():
    
    def __init__(self, n_epochs, learning_rate=0.1):
        
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        
    
    def fit(self, X, y, verbose=False, verbose_debug=False, plot_result_epoch=False):
        
        vect_weihts_start = [random.random() for i in range(len(X.columns))]
        vect_weihts_start.insert(0, -1) # w0
        
        X.insert(0, 'x0', -1)
        
        df = X.reset_index(drop=True).copy()
        df['target'] = y
        self.qtd_classes = len(df['target'].unique())
        
        dict_df = utils.target_one_vs_all(df, col_name_target='target') # 1-out-of-c: pegando dataframes por classe, exemplo: setosa vs outras
        
        self.list_neuron = []
        
        for idx in sorted(dict_df.keys()):
            
            df_classe = dict_df[idx]
            
            neuron = Neuron(self.n_epochs, self.learning_rate)
            
            if plot_result_epoch: print(f'Neuronio {idx}')
            
            self.list_neuron.append(
                
                neuron.fit(df_classe.drop(['target'], axis=1).copy(), df_classe['target'], vect_weihts_start, 
                           verbose=verbose, verbose_debug=verbose_debug, plot_result_epoch=plot_result_epoch)
            
            )
            
            
    def predict(self, X):
        
        X.insert(0, 'x0', -1)
        
        df_test = X.reset_index(drop=True).copy()
        
        list_predict = []
        
        for index, row in df_test.iterrows():
            
            vect_input = df_test.iloc[index, :].tolist()
                        
            list_result=[]
            
            for neuron, n in zip(self.list_neuron, range(self.qtd_classes)):
                
                label, u = neuron.predict(vect_input)
                
                list_result.append((n, label, u))
                            
            df_result_row = pd.DataFrame(list_result, columns=['neuron', 'label', 'u'])
            
            if (df_result_row['label'].sum() > 1) or (df_result_row['label'].sum() == 0): # Mais de 1 neuronio acendeu, ou nenhum acendeu
               
                df_result_row = df_result_row.sort_values(by='u', ascending=False)#Ordenando pelo maior u
                df_result_row.iloc[:, 1] = 0 
                df_result_row.iloc[df_result_row.index.tolist()[0], 1] = 1              
            
            
            list_predict.append(df_result_row['label'].tolist().index(1))
            
        return list_predict