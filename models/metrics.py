from math import sqrt
import numpy as np

def confusion_matrix(y_true, y_pred):
    
    list_class = list(set(y_true))
    matrix_confusion = np.zeros((len(list_class), len(list_class)))
    
    for desejado, predito in zip(y_true,y_pred):
        matrix_confusion[desejado][predito] +=1                
            
    return matrix_confusion


def taxa_acerto(y_true, y_pred, from_mtx_confusion=False): # Taxa de acerto
    
    if from_mtx_confusion:
        
        mtx_confusion = confusion_matrix(y_true, y_pred)
        total = np.sum(mtx_confusion)
        total_acerto = 0
        for class_ in list(set(y_true)):
            total_acerto += mtx_confusion[class_][class_] # Pegando a diagonal
        
        return total_acerto / total
      
    else:
    
        qtd_acertos = 0
        for true, pred in zip(y_true, y_pred):
            if true == pred: 
                qtd_acertos += 1

        return qtd_acertos / len(y_true)
    
def std(lista):
    
    mean = sum(lista) / len(lista)

    variation = sum([(num-mean) ** 2 for num in lista]) / (len(lista) - 1) 

    return sqrt(variation)


def mse(y_true, y_pred):
    
    #list_error = [ ((d ** 2) - (2*y) + (y**2)) for d, y in zip(y_true, y_pred)]
    list_error = [ (d - y) ** 2  for d, y in zip(y_true, y_pred)]
    
    return sum(list_error) / len(list_error)

def rmse(y_true=None, y_pred=None, mse=None):
    if mse == None:
        mse = mse(y_true, y_pred)
    return mse ** (1/2) # pegando a raiz quadrada