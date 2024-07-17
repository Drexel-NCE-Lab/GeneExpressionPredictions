
import pandas as pd
import numpy as np
def evaluate_performance(X_test,y_test,model):
    preds = model.predict(X_test)
    overall_performance = np.mean(np.abs(y_test - np.round(preds)))
    gene_by_gene_performance = np.mean(np.abs(y_test - np.round(preds)), axis=0)
    print(overall_performance ," Mean Average Error")
    return pd.DataFrame(gene_by_gene_performance, columns=['Performance'])