import pandas as pd
import numpy as np
import random
from models import plots, utils
from models.classifiers.PerceptronGeneric import *
        


class Perceptron_Network_Generic():
    
    def __init__(self, n_epochs, learning_rate=0.1, func_ativacao_personalizada=None):
        
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.name_func_ativacao = func_ativacao_personalizada
        
    
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
            
            neuron = Perceptron_Generic(n_epochs=self.n_epochs, 
                                       learning_rate=self.learning_rate,
                                       func_ativacao_personalizada=self.name_func_ativacao)
            
            if plot_result_epoch: print(f'Neuronio {idx}')
            
            self.list_neuron.append(
                
                neuron.fit(df_classe.drop(['target'], axis=1).copy(), df_classe['target'], vect_weihts_start=vect_weihts_start, 
                           verbose=verbose, verbose_debug=verbose_debug, plot_result_epoch=plot_result_epoch)
            
            )
            
            
    def predict(self, X):
        
        X.insert(0, 'x0', -1)
        
        df_test = X.reset_index(drop=True).copy()
        
        list_predict = []
        
        for index, row in df_test.iterrows():
            
#             vect_input = df_test.iloc[index, :].tolist()
            
#             df_input = pd.DataFrame(vect_input, columns=['neuron', 'label', 'u'])
            
            list_result=[]

            df_input = df_test.iloc[[index]]
            
            for neuron, n in zip(self.list_neuron, range(self.qtd_classes)):
                
                label, u = neuron.predict(df_input.copy())
                
                list_result.append((n, label, u))
                            
            df_result_row = pd.DataFrame(list_result, columns=['neuron', 'label', 'u'])
            
            if (df_result_row['label'].sum() > 1) or (df_result_row['label'].sum() == 0): # Mais de 1 neuronio acendeu, ou nenhum acendeu
               
                df_result_row = df_result_row.sort_values(by='u', ascending=False)#Ordenando pelo maior u
                df_result_row.iloc[:, 1] = 0 
                df_result_row.iloc[df_result_row.index.tolist()[0], 1] = 1              
            
            
            list_predict.append(df_result_row['label'].tolist().index(1))
            
        return list_predict