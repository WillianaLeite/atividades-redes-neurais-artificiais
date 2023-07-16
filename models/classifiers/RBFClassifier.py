import pandas as pd
import numpy as np
import random


def process(y):
    if y == 0:
        return [1, 0, 0]
    if y == 1:
        return [0, 1, 0]
    if y == 2:
        return [0, 0, 1]

def back(y):
    if y == [1, 0, 0]:
        return 0
    if y == [0, 1, 0]:
        return 1
    if y == [0, 0, 1]:
        return 2

def process_one_class(y):
    if y == 0:
        return [1, 0]
    if y == 1:
        return [0, 1]

def back_one_class(y):
    if y == [1, 0]:
        return 0
    if y == [0, 1]:
        return 1

def process_six_class(y):
    if y == 0:
        return [1, 0, 0, 0, 0, 0]
    if y == 1:
        return [0, 1, 0, 0, 0, 0]
    if y == 2:
        return [0, 0, 1, 0, 0, 0]
    if y == 3:
        return [0, 0, 0, 1, 0, 0]
    if y == 4:
        return [0, 0, 0, 0, 1, 0]
    if y == 5:
        return [0, 0, 0, 0, 0, 1]

def back_six_class(y):
    if y == [1, 0, 0, 0, 0, 0]: return 0
    if y == [0, 1, 0, 0, 0, 0]: return 1
    if y == [0, 0, 1, 0, 0, 0]: return 2
    if y == [0, 0, 0, 1, 0, 0]: return 3
    if y == [0, 0, 0, 0, 1, 0]: return 4
    if y == [0, 0, 0, 0, 0, 1]: return 5


class RBFClassifier():

    def __init__(self, qt_neurons_hide=5, sigma=2, qt_classes=3, func_ativacao=None):

        self.qt_neurons_hide = qt_neurons_hide
        self.sigma = sigma
        self.qt_classes = qt_classes
        self.func_ativacao=func_ativacao
            

    def logistic(u):
        return 1.0 / (1.0 + np.exp(-u))

    def tanh(u):
        return (1 - np.exp(-u)) / (1 + np.exp(-u))
    
    def degrau(u, threshold=0):
        if u >= threshold: return 1
        else: return 0

    def get_optimal_weigths(df_train, col_target='target'):
        '''
        Função da OLAM
        '''
        X = np.asmatrix(df_train.drop([col_target], axis=1).values)
        D = np.asmatrix(np.array(df_train[col_target].tolist()))

        try:
            W = ((X.T * X).I * X.T * D)
        except:
            W = np.linalg.pinv(X) * D
        return W

    def predict_row(row, W, qt_classes=3, func_ativacao=None):
        '''
        Função da OLAM
        '''
        result = (row.tolist() * W).tolist()[0]
        if func_ativacao is None or func_ativacao == 'logistic':
            result_ = [RBFClassifier.logistic(y) for y in result]
        elif func_ativacao == 'tangente':
            result_ = [RBFClassifier.tanh(y) for y in result]
        if qt_classes <= 2:
            df = pd.DataFrame(zip([process_one_class(i) for i in range(qt_classes)], result_), columns=['class', 'prob'])
            output = back_one_class(df.sort_values(by='prob', ascending=False).iloc[0][0])
        elif qt_classes == 6:
            df = pd.DataFrame(zip([process_six_class(i) for i in range(qt_classes)], result_), columns=['class', 'prob'])
            output = back_six_class(df.sort_values(by='prob', ascending=False).iloc[0][0])
        else:
            df = pd.DataFrame(zip([process(i) for i in range(qt_classes)], result_), columns=['class', 'prob'])
            output = back(df.sort_values(by='prob', ascending=False).iloc[0][0])
        return output


    def predict_dataframe(df, W, qt_classes=3, func_ativacao=None):
        '''
        Função da OLAM
        '''
        list_result = []
        for idx, _ in df.iterrows():
            list_result.append(RBFClassifier.predict_row(df.iloc[idx], W, qt_classes=qt_classes, func_ativacao=func_ativacao))

        return list_result
    
    
    def fit(self, X_train, y_train):
        
        if self.qt_classes <= 2:
            y_train = [process_one_class(y) for y in y_train]
        elif self.qt_classes == 6:
            y_train = [process_six_class(y) for y in y_train]
        else:
            y_train = [process(y) for y in y_train]
        
        df_train = X_train.copy()
        df_train['target'] = y_train

        self.padroes_centro = np.random.uniform(size=(self.qt_neurons_hide, len(X_train.columns))) #row, col
        list_h = []
        for idx, row in X_train.iterrows():

            x = X_train.iloc[idx].tolist()
            
            h = []
            for padrao in self.padroes_centro:
                sub = np.subtract(np.array(x), np.array(padrao))
                calc = np.exp(-(np.dot(np.array(sub).T, np.array(sub)) / 2 * (self.sigma ** 2)))
                h.append(calc)

            list_h.append(h)

        M = np.asmatrix(list_h)
        input_olam = np.insert(M, 0, -1, axis=1)
        df_input_olam = pd.DataFrame(input_olam, columns = [f'h{i}' for i in range(input_olam.shape[1])])
        df_input_olam = pd.DataFrame(input_olam, columns = [f'h{i}' for i in range(input_olam.shape[1])])
        df_input_olam['target'] = df_train['target']

        self.weigths = RBFClassifier.get_optimal_weigths(df_input_olam, col_target='target')

    
    def predict(self, X_test):

        list_h_test = []
        for idx, row in X_test.iterrows():

            x = X_test.iloc[idx].tolist()
            
            h = []
            for padrao in self.padroes_centro:
                
                sub = np.subtract(np.array(x), np.array(padrao))
                calc = np.exp(-(np.dot(np.array(sub).T, np.array(sub)) / 2 * (self.sigma ** 2)))
                h.append(calc)

            list_h_test.append(h)

        M_teste = np.asmatrix(list_h_test)
        input_olam_test = np.insert(M_teste, 0, -1, axis=1)
        df_input_olam_test = pd.DataFrame(input_olam_test, columns = [f'h{i}' for i in range(input_olam_test.shape[1])])
        
        return RBFClassifier.predict_dataframe(df_input_olam_test, self.weigths, self.qt_classes, self.func_ativacao)
