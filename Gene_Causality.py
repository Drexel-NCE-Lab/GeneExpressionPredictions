import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack, vstack, csr_matrix
import numpy as np
from scipy.io import savemat, mmread
import math
from boruta import BorutaPy
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor 
import pickle

class GeneCasuality_Search:
    def __init__(self):
        self.tf_masterlist = []
        self.cam_masterlist = []
        self.tf_data = None
        self.cam_data = None
        self.working_metadata = None
        self.__allmetadata = None
        self.__alldata = None
        self.__genenames = None
        self.working_genes = None
        self.working_data = None
        self.trained_model = None
        self.train = None
        self.test = None
        self.gene_performance = None

    def __str__(self):
        if self.tf_data is not None and self.cam_data is not None:  # Changed condition to check if data is not None
            working_metadata = self.working_metadata.copy().reset_index(drop=True)
            # Concatenate with current gene metadata
            working_df = pd.concat([working_metadata, pd.DataFrame(self.working_data.toarray(), columns=list(self.working_genes.gene_name))], axis=1)
            return working_df.head().to_string()  # Convert DataFrame to string for printing
        else:
            return "TF or CAM data is not available."

    def reset_analysis(self):
        # Clearing properties for reset
        self.working_metadata = None
        self.working_genes = None
        self.working_data = None
        self.trained_model = None
        # Resetting to original data copies
        self.working_genes = self.__genenames.copy() if self.__genenames is not None else None
        self.working_data = self.__alldata.copy() if self.__alldata is not None else None
        self.working_metadata = self.__allmetadata.copy() if self.__allmetadata is not None else None

    def read_genomic_data(self, early_path, main_path):
        early = mmread(early_path).transpose().astype(np.uint8).tocsr()
        main = mmread(main_path).transpose().astype(np.uint8).tocsr()
        self.__alldata = vstack([main, early])
        self.working_data = self.__alldata.copy()

    def read_genes(self, feature_path):
        all_info = pd.read_csv(feature_path, names=['fbname', 'gene_name', 'expression'], delimiter='\t')
        self.__genenames = all_info
        self.working_genes = all_info.copy()

    def subset_tf_cam(self):
        tf_indices = list(self.__genenames[self.__genenames['gene_name'].str.lower().isin(self.tf_masterlist)].index)
        cam_indices = list(self.__genenames[self.__genenames['gene_name'].str.lower().isin(self.cam_masterlist)].index)
        master_indices = tf_indices + cam_indices
        self.tf_data = self.working_data[:, tf_indices]
        self.cam_data = self.working_data[:, cam_indices]
        self.working_data = self.working_data[:, master_indices]
        self.working_genes = self.__genenames.iloc[master_indices, :]

    def read_metadata(self, early_metadata_path, main_metadata_path, feature_path):
        metadata_main = pd.read_csv(main_metadata_path, sep='\t')
        metadata_early = pd.read_csv(early_metadata_path, sep='\t')
        genes = pd.read_csv(feature_path, sep='\t', names=['gene_id', 'gene_name', 'analysis'])
        metadata_main.set_index('barcode', inplace=True)
        metadata_early.set_index('barcode', inplace=True)
        metadata_combined = pd.concat([metadata_main, metadata_early]).reset_index()
        self.__allmetadata = metadata_combined
        self.working_metadata = metadata_combined.copy()

    def read_master_genes(self, tf_path, cam_path):
        tf_list = pd.read_csv(tf_path).iloc[:, 0].str.lower().tolist()
        cam_list = pd.read_csv(cam_path).iloc[:, 0].str.lower().tolist()
        self.cam_masterlist += cam_list
        self.tf_masterlist += tf_list
        common_genes = set(self.cam_masterlist) & set(self.tf_masterlist)
        if common_genes:
            print('Found gene_names which belonged to both CAM & TF:')
            for x in common_genes:
                print(x)
            raise ValueError('Duplicate gene references found.')

    def cross_validate(self, test_size, num_folds, params, num_boost_rounds, early_stopping_rounds):
        X_train, X_test, y_train, y_test = train_test_split(self.tf_data, self.cam_data, test_size=test_size)
        dtrain = xgb.DMatrix(X_train, label=y_train.toarray())
        cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=num_folds,
                            num_boost_round=num_boost_rounds,
                            early_stopping_rounds=early_stopping_rounds,
                            metrics='rmse', as_pandas=True)
        return cv_results

    def train_model(self, test_size, params):
        X_train, X_test, y_train, y_test = train_test_split(self.tf_data, self.cam_data, test_size=test_size)
        dtrain = xgb.DMatrix(X_train, label=y_train.toarray())
        dtest = xgb.DMatrix(X_test, label=y_test.toarray())
        self.trained_model = xgb.train(params, dtrain,num_boost_round = params.get('n_estimators',100))
        self.train = dtrain
        self.test = (dtest, y_test)

    def evaluate_performance(self):
        preds = self.trained_model.predict(self.test[0])
        overall_performance = np.mean(np.abs(self.test[1].toarray() - np.round(preds)))
        gene_by_gene_performance = np.mean(np.abs(self.test[1].toarray() - np.round(preds)), axis=0)
        return overall_performance, pd.DataFrame(gene_by_gene_performance, columns=['Performance'])

    def reduce_feature(self):
        rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
        X_filtered = feat_selector.fit_transform(self.tf_data.toarray(), self.cam_data.toarray())
        return X_filtered
    def grid_search(self,param_grid):
        X_train, X_test, y_train, y_test = train_test_split(self.tf_data, self.cam_data, test_size=5e-2)
        xgb_model = XGBRegressor(tree_method='gpu_hist', objective='reg:squarederror', verbosity=2)
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3, verbose=2)
        grid_search.fit(X_train, y_train)
        return pd.DataFrame(grid_search.cv_results_)
    def save_model(path):
        pass
    def subset_data(self, **kwargs):
        searching = self.__allmetadata.copy()
        for key, val in kwargs.items():
            if isinstance(val, list):
                searching = searching[searching[key].isin(val)]
            else:
                searching = searching[searching[key] == val]
        rows = list(searching.index)
        self.working_metadata = searching
        self.working_data = self.working_data[rows, :].copy()
