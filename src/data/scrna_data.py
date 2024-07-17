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
from scipy.sparse import csr_matrix,vstack,save_npz,load_npz
from xgboost import XGBRegressor 
import pickle

class scrna:
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
        self.gene_column = None
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

    def read_genomic_data(self,paths, rows='',sample_cutoff = 200):
        """
        Must be of type .npz matrices. 

        Added a row parameter which specifies if genes or samples are on the rows
        """
        if type(paths) is not list:
            paths = [paths]
        expression = None
        for path in paths:
            expression = load_npz(path) if expression is None else vstack([expression, load_npz(path)])
        
        if rows == 'genes':
            pre_cut = expression.T.tocsr()
        else:
            pre_cut = expression.tocsr()

        col_sums = pre_cut.sum(axis=1)
        

        self.__alldata = pre_cut 
        self.working_data = self.__alldata.copy()

    def read_genes(self, feature_path,gene_column='Genes',column_names = []):
        """
        Needs to be either a csv or tsv. OR pkl file
        """
        file_type = feature_path.split('.')[-1]
        if file_type == 'pkl': 
            with open(feature_path,'rb') as f:
                all_info = pd.DataFrame({gene_column:pickle.load(f)})
                if len(all_info.columns) == len(column_names):
                    all_info.columns = column_names
        elif file_type =='tsv':
            all_info = pd.read_csv(feature_path, delimiter='\t',names=column_names)
        else:
            all_info = pd.read_csv(feature_path,names=column_names)
        self.gene_column = gene_column
        self.__genenames = all_info
        self.working_genes = all_info.copy()
    def subset_tf_cam(self):
        tf_indices = list(self.__genenames[self.__genenames[self.gene_column].str.lower().isin(self.tf_masterlist)].index)
        cam_indices = list(self.__genenames[self.__genenames[self.gene_column].str.lower().isin(self.cam_masterlist)].index)
        master_indices = tf_indices + cam_indices
        self.tf_data = self.working_data[:, tf_indices]
        self.cam_data = self.working_data[:, cam_indices]
        self.working_data = self.working_data[:, master_indices]
        self.working_genes = self.__genenames.iloc[master_indices, :]
    def read_metadata(self, metadata_path):
        metadata = pd.read_csv(metadata_path)
        
        self.__allmetadata = metadata
        self.working_metadata = metadata.copy()
    def read_master_genes(self, tf_path, cam_path):

        """
        This will read in the list of genes whihc belong to categories Transcription Factor and Cellular Adhesion Molecule. 

        TF path goes FIRST and CAM Path goes SECOND
        """
        self.cam_masterlist = pd.read_csv(cam_path).iloc[:, 0].str.lower().tolist()
        self.tf_masterlist = pd.read_csv(tf_path).iloc[:, 0].str.lower().tolist()
        common_genes = set(self.cam_masterlist) & set(self.tf_masterlist)
        if common_genes:
            print('Found gene_names which belonged to both CAM & TF:')
            for x in common_genes:
                print(x)
            raise ValueError('Duplicate gene references found.')
    def subset_data(self, **kwargs):
        print(kwargs)
        searching = self.__allmetadata.copy()
        for key, val in kwargs.items():
            if isinstance(val, list):
                searching = searching[searching[key].isin(val)]
            else:
                searching = searching[searching[key] == val]
        rows = list(searching.index)
        print(rows)
        self.working_metadata = searching
        self.working_data = self.working_data[rows, :].copy()
    def get_current_data(self):
        return pd.DataFrame(self.working_data.toarray(), columns=list(self.working_genes[self.gene_column]))