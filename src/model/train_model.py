import xgboost as xgb

def train_model(X_train,y_train, params = {}):
   
    dtrain = xgb.DMatrix(X_train, label=y_train.toarray())
    
    model = xgb.train(params, dtrain,num_boost_round = params.get('n_estimators',100))
    return model

