import pandas as pd
import numpy as np
import random

class RBFRegressor():

    def __init__(self, n_neuron_hide=5, sigma=2, qt_classes=3):

        self.q = n_neuron_hide
        self.sigma = sigma
        self.qt_classes = qt_classes

    def logistic(u):
        return 1.0 / (1.0 + np.exp(-u))

    def tanh(u):
        return (1 - np.exp(-u)) / (1 + np.exp(-u))
    
    def degrau(u, threshold=0):
        if u >= threshold: return 1
        else: return 0

    def get_optimal_weigths(df_train, col_target='target', qt_classes=3):
        '''
        Função da OLAM
        '''
        X = np.asmatrix(df_train.drop([col_target], axis=1).values)
        D = np.asmatrix(np.array(df_train[col_target].tolist()))
        W = np.linalg.pinv(X) * D

        return W

    def predict_row(row, W, qt_classes=3):
        '''
        Função da OLAM
        '''
        return np.dot(row, W)


    def predict_dataframe(df, W, qt_classes=3):
        '''
        Função da OLAM
        '''
        list_result = []
        for idx, _ in df.iterrows():
            list_result.append(RBFRegressor.predict_row(df.iloc[idx], W, qt_classes=qt_classes))

        return list_result
    
    
    def fit(self, X_train, y_train):

        df_train = X_train.copy()
        df_train['target'] =[[y] for y in y_train]

        self.padroes_centro = np.random.uniform(size=(self.q, len(X_train.columns))) #row, col
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
        df_input_olam['target'] = df_train['target']

        self.weigths = RBFRegressor.get_optimal_weigths(df_input_olam, col_target='target', qt_classes=self.qt_classes)

    
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
        
        X_test['predict'] = RBFRegressor.predict_dataframe(df_input_olam_test, self.weigths, self.qt_classes)
        X_test['predict'] = X_test['predict'].astype(str)
        X_test['predict'] = X_test['predict'].apply(lambda x: str(str(x).replace('[', '').replace(']', '')))
        X_test['predict'] = X_test['predict'].astype(float)

        return X_test['predict']