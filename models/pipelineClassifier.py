import pandas as pd
import numpy as np
import random
from models import utils, metrics
from models.classifiers.Perceptron import *
from models.classifiers.PerceptronNetworkGeneric import *
from models.classifiers.PerceptronGeneric import *
from models.classifiers.MLPClassifier import *
from models.classifiers.RBFClassifier import *


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
            if param in ['learning_rate']: # Parâmetros aceitos
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
            
                list_taxa_acerto = []
                for train_index, valid_index in k_fold.split(X):

                    X_train_fold, X_valid_fold = df_train.drop(['target'], axis=1).iloc[train_index].copy(), df_train.drop(['target'], axis=1).iloc[valid_index].copy()
                    y_train_fold, y_valid_fold = df_train.iloc[train_index]['target'], df_train.iloc[valid_index]['target']
                    
                    if param in ['learning_rate']:
                        self.classifier.learning_rate = param_value
                    elif param in ['qt_neurons_hide']:
                        self.classifier.qt_neurons_hide = param_value
                    elif param in ['sigma']:
                        self.classifier.sigma = param_value
                    
                    
                    self.classifier.fit(X_train_fold, y_train_fold)
                    
                    X_valid_fold['predict'] = self.classifier.predict(X_valid_fold)
                    X_valid_fold['y'] = y_valid_fold
                    taxa_acerto = metrics.taxa_acerto(X_valid_fold['y'], X_valid_fold['predict'])
                    list_taxa_acerto.append(taxa_acerto)
                
                taxa_acerto_media = sum(list_taxa_acerto) / self.k_fold

                dict_info[param_value] = taxa_acerto_media
                
            dict_return[param] = max(dict_info, key=dict_info.get) #taxa de aprendizado que teve maior média de taxa de acerto

        return dict_return

class make_pipeline():
    
    def __init__(self, model_name='Perceptron', n_realizations=10, task='binary_classification', qtd_classes=None, func_ativacao_personalizada=None):
        self.model_name = model_name
        self.n_realizations = n_realizations
        self.task = task
        self.qtd_classes = qtd_classes
        if self.task == 'multiclass_classification':
            if func_ativacao_personalizada is None: self.func_ativacao ='degrau'
            else: self.func_ativacao = func_ativacao_personalizada
        if self.task == 'binary_classification':
            self.func_ativacao = func_ativacao_personalizada
        
    def __select_realization(self, accuracy, list_taxa_acertos, dict_realizations):
        
        min_dist_mean = 1000000000
        n_realization_select = -1
        for n_realization, dist_mean in zip(dict_realizations.keys(), [abs(num-accuracy) for num in list_taxa_acertos]):
            
            if dist_mean < min_dist_mean: 
                min_dist_mean = dist_mean
                n_realization_select = n_realization
                
        return dict_realizations[n_realization_select]
    
    def __get_model(self, n_epochs):
        
        if self.model_name == 'Perceptron':
            if self.task == 'binary_classification':
                perceptron = Perceptron_Generic(n_epochs=n_epochs, func_ativacao_personalizada=self.func_ativacao)
            elif self.task == 'multiclass_classification':
                perceptron = Perceptron_Network_Generic(n_epochs=n_epochs, func_ativacao_personalizada=self.func_ativacao)
            return perceptron
        elif self.model_name == 'MLP':
            if self.task == 'binary_classification':
                mlp = MLPClassifier(
                    n_epochs = n_epochs, 
                    func_ativacao = self.func_ativacao)
                return mlp
        
        elif self.model_name == 'RBF':
            rbf = RBFClassifier(qt_classes = self.qtd_classes, func_ativacao=self.func_ativacao)
            return rbf
        
    def run_realizations(self, df, k_fold, param_grid, col_target='target', train_size=0.8, stratify=True, 
                         n_epochs=10, normalize=True):
        
        dict_realizations = {}
        
        list_taxa_acertos = []
        
        for realization in range(self.n_realizations):
            
            print('\n\n')
            print('-' *125)
            print(f'Realização: {realization}')
            
            if normalize:
                for col in df.columns:
                    if col != col_target:
                        df = utils.normalize_col(df, col)
                
            df = df.sample(frac=1).reset_index(drop=True) # Shuffle

            X_train, y_train, X_test, y_test = utils.split_train_test(df, col_target, train_size=train_size, 
                                                                      stratify=stratify)
            model = self.__get_model(n_epochs)
                    
            print(f'Start Grid Search k-fold: {k_fold}')
            grid_search = GridSearchCV(param_grid=param_grid, classifier=model, k_fold=k_fold)

            dict_best_param = grid_search.fit(X_train, y_train)
            
            if 'learning_rate' in dict_best_param.keys():
                model.learning_rate = dict_best_param['learning_rate']
            if 'qt_neurons_hide' in dict_best_param.keys():
                model.qt_neurons_hide = dict_best_param['qt_neurons_hide']
            if 'sigma' in dict_best_param.keys():
                model.sigma = dict_best_param['sigma']    
            

            model.fit(X_train, y_train)

            # Teste
            df_test = X_test.copy()
            df_test['predict'] = model.predict( df_test )
            df_test['target'] = y_test
            
            if self.task == 'binary_classification':
            
                confusion_matrix = metrics.confusion_matrix(df_test['target'], df_test['predict'])
                taxa_acerto = metrics.taxa_acerto(df_test['target'], df_test['predict'], from_mtx_confusion=False)

                df_train = X_train.copy()
                df_train['target'] = y_train

                dict_realizations[realization] = {

                    'taxa_acerto': taxa_acerto,
                    'confusion_matrix': confusion_matrix,
                    'classifier': model,
                    'train_data': df_train,
                    'test_data': df_test
                }
                
                print('Confusion Matrix: ')
                print(confusion_matrix)
                
            elif self.task == 'multiclass_classification':
                taxa_acerto = metrics.taxa_acerto(df_test['target'], df_test['predict'], from_mtx_confusion=False)

                df_train = X_train.copy()
                df_train['target'] = y_train

                dict_realizations[realization] = {

                    'taxa_acerto': taxa_acerto,
                    'classifier': model,
                    'train_data': df_train,
                    'test_data': df_test
                }
             
            
            list_taxa_acertos.append(taxa_acerto)
            
            print(f'Taxa de Acerto: {taxa_acerto}')
            print('-' *125)
            
        accuracy = sum(list_taxa_acertos) / self.n_realizations

        desvio_padrao = metrics.std(list_taxa_acertos)
        
        # Selecionando realização que teve taxa de acerto que mais se aproximou da acurácia 
        realization_select = self.__select_realization(accuracy, list_taxa_acertos, dict_realizations)

        return accuracy, desvio_padrao, realization_select, dict_realizations