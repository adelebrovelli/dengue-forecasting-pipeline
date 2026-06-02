import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def medape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    medape = np.median(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
    return medape

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
    return mape

def auc(y_true, y_pred):
    try :
        return roc_auc_score(y_true, y_pred)
    except:
        return np.nan

def f1_score(y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(len(y_true)):    
        if y_true[i] == 1 and y_pred[i] == 1:
            tp+=1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp+=1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn+=1
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn+=1 


    if (tp+fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp) 

    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn) 

    if (precision + recall == 0):
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return f1