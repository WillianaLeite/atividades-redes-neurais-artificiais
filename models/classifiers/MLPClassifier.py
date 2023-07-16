import pandas as pd
import numpy as np
import random

class MLPClassifier():

      def __init__(self, n_epochs, qt_neurons_hide=3, qt_neuron_output=1, learning_rate=0.1, func_ativacao=None):
          self.n_epochs = n_epochs
          self.learning_rate = learning_rate
          self.qt_neurons_hide = qt_neurons_hide
          self.qt_neuron_output = qt_neuron_output
          if func_ativacao == 'logistic':
              self.func_ativacao = self.logistic
              self.func_deriv_ativacao = self.logistic_deriv
          elif func_ativacao == 'tangente':
              self.func_ativacao = self.tanh
              self.func_deriv_ativacao = self.tanh_deriv

      def logistic(self, u):
        return 1.0 / (1.0 + np.exp(-u))

      def logistic_deriv(self, Y):
          return Y * (1.0 - Y)

      def tanh(self, u):
          return (1 - np.exp(-u)) / (1 + np.exp(-u))

      def tanh_deriv(self, Y):
          return 1.0 - (Y**2)
      
      def degrau(self, u, threshold=0, verify=True):
          if verify:
              if self.func_ativacao == self.logistic:
                  threshold = 0.5
          if u >= threshold: return 1
          else: return 0

      def __calc_u(self, vect_input, vect_weigths):
          return np.dot(vect_input, vect_weigths)

      def fit(self, X_train, y_train):

          self.qt_neuron_input = len(X_train.columns)

          # Criando pesos aleatórios
          self.weights_hide = np.random.uniform(size=(self.qt_neuron_input, self.qt_neurons_hide))
          self.weights_hidde_bias = np.random.uniform(size=(1, self.qt_neurons_hide))
          self.weights_output = np.random.uniform(size=(self.qt_neurons_hide, self.qt_neuron_output))
          self.weights_output_bias = np.random.uniform(size=(1, self.qt_neuron_output))

          df_train = X_train.copy()
          df_train['target'] = y_train
          X = df_train.drop(['target'], axis=1).values
          desejavel = df_train[['target']].values
          list_error = []
          for _ in range(self.n_epochs):
              
              #Foward
              vec_u_hide = self.__calc_u(X, self.weights_hide) 
              vec_u_hide += self.weights_hidde_bias
              output_layer_hide = self.func_ativacao(vec_u_hide)

              vec_u_output = self.__calc_u(output_layer_hide, self.weights_output) 
              vec_u_output += self.weights_output_bias
              predict = self.func_ativacao(vec_u_output)

              #Backword
              error = desejavel - predict
              list_error.append(sum(error))
              deriv_predict = error * self.func_deriv_ativacao(predict) # ej * y'
              
              error_hide = deriv_predict.dot(self.weights_output.T) #(mji * ej * y')
              deriv_hide = error_hide * self.func_deriv_ativacao(output_layer_hide) # mji * ej * y' * hi'

              #Atualizando todos os pesos
              self.weights_output += output_layer_hide.T.dot(deriv_predict) * self.learning_rate
              self.weights_output_bias += np.sum(deriv_predict,axis=0, keepdims=True) * self.learning_rate
              self.weights_hide += X.T.dot(deriv_hide) * self.learning_rate
              self.weights_hidde_bias += np.sum(deriv_hide,axis=0, keepdims=True) * self.learning_rate
          
          return list_error

      def predict(self, X_test):

          list_predict = []
          
          X = X_test.values
          
          # Foward
          vec_u_hide = self.__calc_u(X, self.weights_hide)
          vec_u_hide += self.weights_hidde_bias
          output_layer_hide = self.func_ativacao(vec_u_hide)

          vec_u_output = self.__calc_u(output_layer_hide, self.weights_output) 
          vec_u_output += self.weights_output_bias
          predict = self.func_ativacao(vec_u_output)

          return list(map(self.degrau, predict))