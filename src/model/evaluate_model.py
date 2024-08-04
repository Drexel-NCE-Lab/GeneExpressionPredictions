
import pandas as pd
import numpy as np
import xgboost as xgb
def evaluate_performance(X_test,y_test,model):
    X_test_vals = X_test.values
    y_test_vals = y_test.values
    dtrain = xgb.DMatrix(X_test_vals, label=y_test_vals)
    preds = model.predict(dtrain)
    overall_performance = np.mean(np.abs(y_test - np.round(preds)))
    gene_by_gene_performance = np.mean(np.abs(y_test - np.round(preds)), axis=0)
    print(overall_performance ," Mean Average Error")
    return pd.DataFrame(gene_by_gene_performance, columns=['Performance'])