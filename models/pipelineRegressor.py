import pandas as pd
import numpy as np
import random
from models import utils, metrics
from models.regressors.AdalineRegressor import *
from models.regressors.RBFRegressor import *

class CrossValidationKFold():
    
    def __init__(self, k_fold):
        
        self.__current_k = 0
        self.k_fold = k_fold
    
    def split(self, X):
        
        self.X = X
        
        dict_idx_folds = {}
        
        black_list = []
        
        list_index = self.X.index.values.tolist()
        
        n_samples_fold = round(len(X) / self.k_fold)
        
        index = 0
        idx_start = 0 
        for k in range(n_samples_fold, len(X) + 1, n_samples_fold):
            
            dict_idx_folds[index] = list_index[idx_start: k]
            idx_start = k
            index+=1
        
        list_return = []
        for k in reversed(list(dict_idx_folds.keys())): # K descrescente
            
            idx_test = dict_idx_folds[k]
            
            idx_train = []
            
            for k_train in list(dict_idx_folds.keys()):
                
                if k_train != k:
                        
                    idx_train += dict_idx_folds[k_train]
                    
            list_return.append((idx_train, idx_test))
       
        return list_return

    
class GridSearchCV(): # CV: CrossValidation
    
    def __init__(self, classifier, param_grid, k_fold=5): 
        self.classifier = classifier
        
        self.param_grid = {}
        for param in param_grid.keys():
            if param in ['learning_rate']: # ParÃ¢metros aceitos
                self.param_grid[param] = param_grid[param]
            
        self.k_fold = k_fold
        
    def fit(self, X, y):
        
        dict_return = {}
        
        for param, list_param_value in self.param_grid.items():
            
            k_fold = CrossValidationKFold(self.k_fold)
            
            df_train = X.copy()
            df_train['target'] = y
            dict_info = {}
            
            for param_value in list_param_value:
            
                list_mse = []
                for train_index, valid_index in k_fold.split(X):

                    X_train_fold, X_valid_fold = df_train.drop(['target'], axis=1).iloc[train_index].copy(), df_train.drop(['target'], axis=1).iloc[valid_index].copy()
                    y_train_fold, y_valid_fold = df_train.iloc[train_index]['target'], df_train.iloc[valid_index]['target']
                    
                    if param in ['learning_rate']:
                        self.classifier.learning_rate = param_value
                    if param in ['qt_neurons_hide']:
                        self.classifier.qt_neurons_hide = param_value
                    if param in ['sigma']:
                        self.classifier.sigma = param_value
                    
                    self.classifier.fit(X_train_fold, y_train_fold)
                    
                    X_valid_fold['predict'] = self.classifier.predict(X_valid_fold)
                    X_valid_fold['y'] = y_valid_fold

                    mse = metrics.mse(X_valid_fold['y'], X_valid_fold['predict'])
                    list_mse.append(mse)
                
                mse_medio = sum(list_mse) / self.k_fold

                dict_info[param_value] = mse_medio
                
            dict_return[param] = min(dict_info, key=dict_info.get) #Taxa de aprendizdo que teve menor mse

        return dict_return

class make_pipeline():
    
    def __init__(self, n_realizations=10, model_name='Adaline'):
        self.n_realizations = n_realizations       
        self.model_name = model_name
        
    def get_model(self, n_epochs=None):
        
        if self.model_name == 'Adaline': return AdalineRegressor(n_epochs=n_epochs)
        
        elif self.model_name == 'RBFRegressor': return RBFRegressor(qt_classes=1)
        
    def run_realizations(self, df, k_fold, param_grid, col_target='target', train_size=0.8, stratify=True, 
                         n_epochs=10, normalize=True):
        
        dict_realizations = {}
        
        list_mse = []
        
        list_rmse = []
        
        for realization in range(self.n_realizations):
            
            print('\n\n')
            print('-' *125)
            print(f'RealizaÃ§Ã£o: {realization}')
            
            if normalize:
                for col in df.columns:
                    if col != col_target:
                        df = utils.normalize_col(df, col)
                
            df = df.sample(frac=1).reset_index(drop=True) # Shuffle

            X_train, y_train, X_test, y_test = utils.split_train_test(df, col_target, train_size=train_size, 
                                                                      stratify=stratify)
            
            perceptron = self.get_model(n_epochs=n_epochs)
            
            print(f'Start Grid Search k-fold: {k_fold}')
            grid_search = GridSearchCV(param_grid=param_grid, classifier=perceptron, k_fold=k_fold)

            dict_best_param = grid_search.fit(X_train, y_train)
            
            if 'learning_rate' in dict_best_param.keys():
                print(f'Best learning_rate: {dict_best_param["learning_rate"]}')
                perceptron.learning_rate = dict_best_param['learning_rate']
            if 'qt_neurons_hide' in dict_best_param.keys():
                print(f'Best qt_neurons_hide: {dict_best_param["qt_neurons_hide"]}')
                perceptron.qt_neurons_hide = dict_best_param['qt_neurons_hide']
            if 'sigma' in dict_best_param.keys():
                print(f'Best sigma: {dict_best_param["sigma"]}')
                perceptron.sigma = dict_best_param['sigma']

            perceptron.fit(X_train, y_train)

            # Teste
            df_test = X_test.copy()
            df_test['predict'] = perceptron.predict( df_test )
            df_test['target'] = y_test
            
            mse = metrics.mse(df_test['target'], df_test['predict'])
            rmse = metrics.rmse(mse=mse)
            
            list_mse.append(mse)
            list_rmse.append(rmse)
            
            df_train = X_train.copy()
            df_train['target'] = y_train
            
            dict_realizations[realization] = {
                
                'mse': mse,
                'rmse': rmse,
                'regressor': perceptron,
                'train_data': df_train,
                'test_data': df_test
            }
            
            print(f'MSE: {mse}')
            print(f'RMSE: {rmse}')
            print('-' *125)
            
        mean_mse = sum(list_mse) / self.n_realizations
        
        mean_rmse = sum(list_rmse) / self.n_realizations
        
        desvio_padrao_mse = metrics.std(list_mse)
        
        desvio_padrao_rmse = metrics.std(list_rmse)
        
        return mean_mse, mean_rmse, desvio_padrao_mse, desvio_padrao_rmse, dict_realizations