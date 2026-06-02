import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def bootstrap(y_true, y_pred, metrica, n_iterations=1000, classificacao=False):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    scores = []
    n = len(y_true)
    
    for _ in range(n_iterations):
       
        amostras = np.random.choice(n, size=n, replace=True)
        
        y_true_boot = y_true[amostras]
        y_pred_boot = y_pred[amostras]
        
     
        if classificacao:
            
            if len(np.unique(y_true_boot)) < 2:
                continue
        
    
        try:
            score = metrica(y_true_boot, y_pred_boot)
            scores.append(score)
        except:
            
            continue
    
    scores = np.array(scores)
    
    return {
        'media': np.mean(scores),
        'ic_inferior': np.percentile(scores, 2.5),
        'ic_superior': np.percentile(scores, 97.5),
        'desvio_padrao': np.std(scores)
    }


def validacao_cruzada_temporal(X, y, n_splits):

    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    folds = []
    for i, (train_x, test_x) in enumerate(tscv.split(X)):
        folds.append({
            'fold': i + 1,
            'treino': train_x,
            'teste': test_x,
            'tamanho_treino': len(train_x),
            'tamanho_teste': len(test_x)
        })
    
    return folds