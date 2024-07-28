import xgboost as xgb
import pandas as pd
def train_model(X_train: pd.DataFrame,y_train: pd.DataFrame, params = {}):
    X_train = X_train.values
    y_train = y_train.values
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    model = xgb.train(params, dtrain,num_boost_round = params.get('n_estimators',100))
    return model

