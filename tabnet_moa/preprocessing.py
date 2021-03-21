import random
import os
import torch
import sys
import tqdm
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
os.environ["PYTHONHASHSEED"] = str(42)

class Load:
    def __init__(self, train_features='', train_targets_scored='', test_features='', submission=''):
        '''
        Read CSV files for features and binary MoA targets of train data, features of test data and submission respectively.

        Args:
          train_features (str): file path of features for train data
          train_targets_scored (str): file path of binary MoA targets for train data
          test_features (str): file path of features for test data
          submission (str): file path for submission

        Attributes:
          train (dataframe): input data of features for train data
          targets (dataframe): input data of known responses (binary) from MoA targets for train data
          test (dataframe): input data of features for test data
          submission (dataframe): input data of predicted responses from MoA targets for test data
        '''
        self.train = pd.read_csv(train_features)
        self.targets = pd.read_csv(train_targets_scored)
        self.test = pd.read_csv(test_features)
        self.submission = pd.read_csv(submission)

    def drop_ctl_vehicle(self):
        '''
        Drop samples with control perturbations since control perturbations have no MoAs.

        Returns:
          train (dataframe): updated input data of features for train data
          test (dataframe): updated input data of features for test data
          targets (dataframe): updated input data of known responses (binary) from MoA targets for train data
        '''
        train = self.train[self.train["cp_type"] != "ctl_vehicle"]
        test = self.test[self.test["cp_type"] != "ctl_vehicle"]
        targets = self.targets.iloc[train.index]
        train.reset_index(drop = True, inplace = True)
        test.reset_index(drop = True, inplace = True)
        targets.reset_index(drop = True, inplace = True)

        return train, test, targets

class ScaledPCA:
    def __init__(self, scaler):
        '''
        Load parameters for scaling features and pca in input data.

        Attributes:
          scaler (obj): scaler
          variance_threshold (int): threshold of variance
          ncompo_genes (int): number of principal components for genes
          ncompo_cells (int): number of principal components for cells
        '''
        self.scaler = scaler
        self.variance_threshold = 0.9
        self.ncompo_genes = 80
        self.ncompo_cells = 10

    def run_scaling(self, loaded_train, loaded_test):
        '''
        Transform input data by scaling numeric features.

        Args:
          loaded_train (dataframe): input data of features for train data
          loaded_test (dataframe): input data of features for test data

        Returns:
          data_all (dataframe): transformed input data of features for train and test data
        '''
        data_all = pd.concat([loaded_train, loaded_test], ignore_index=True)
        cols_numeric = [feat for feat in list(data_all.columns) if feat not in ["sig_id", "cp_type", "cp_time", "cp_dose"]]
        mask = (data_all[cols_numeric].var() >= self.variance_threshold).values
        tmp = data_all[cols_numeric].loc[:, mask]
        data_all = pd.concat([data_all[["sig_id", "cp_type", "cp_time", "cp_dose"]], tmp], axis = 1)
        cols_numeric = [feat for feat in list(data_all.columns) if feat not in ["sig_id", "cp_type", "cp_time", "cp_dose"]]
        data_all[cols_numeric] = self.scaler.fit_transform(data_all[cols_numeric])

        return data_all

    def run_pca(self, scaled_data_all):
        '''
        Apply Principal Component Analysis (PCA) on gene expression and cell viability features of the input data respectively.

        Args:
          scaled_data_all (dataframe): input data of features for train and test data

        Returns:
          concat_data_all (dataframe): input data of original features and principal component features for train and test data
        '''
        GENES = [col for col in scaled_data_all.columns if col.startswith("g-")]
        CELLS = [col for col in scaled_data_all.columns if col.startswith("c-")]

        pca_genes = PCA(n_components = self.ncompo_genes, random_state = 42).fit_transform(scaled_data_all[GENES])
        pca_cells = PCA(n_components = self.ncompo_cells, random_state = 42).fit_transform(scaled_data_all[CELLS])

        pca_genes = pd.DataFrame(pca_genes, columns = [f"pca_g-{i}" for i in range(self.ncompo_genes)])
        pca_cells = pd.DataFrame(pca_cells, columns = [f"pca_c-{i}" for i in range(self.ncompo_cells)])
        concat_data_all = pd.concat([scaled_data_all, pca_genes, pca_cells], axis=1)

        return concat_data_all

    def one_hot_encoding(self, concat_data_all):
        '''
        One hot encode the categorical features for treatment time and treatment dosage of the input data respectively.
        Then for each observation, adds the sum, mean, standard deviation, kurtosis and skewedness of cell viability,
        gene expression and combined cell viability and gene expression data respectively as features.

        Args:
          concat_data_all (dataframe): input data of features for train and test data

        Returns:
          encoded_data_all (dataframe): input data of original and new features for train and test data
        '''
        encoded_data_all = pd.get_dummies(concat_data_all, columns = ["cp_time", "cp_dose"])

        encoded_GENES = [col for col in encoded_data_all.columns if col.startswith("g-")]
        encoded_CELLS = [col for col in encoded_data_all.columns if col.startswith("c-")]

        for stats in tqdm.tqdm(["sum", "mean", "std", "kurt", "skew"]):
            encoded_data_all["g_" + stats] = getattr(encoded_data_all[encoded_GENES], stats)(axis = 1)
            encoded_data_all["c_" + stats] = getattr(encoded_data_all[encoded_CELLS], stats)(axis = 1)
            encoded_data_all["gc_" + stats] = getattr(encoded_data_all[encoded_GENES + encoded_CELLS], stats)(axis = 1)

        return encoded_data_all


class Preprocess:
    def __init__(self):
        '''
        Load features to drop from input data.

        Attributes:
          features_to_drop (arr): list of features to drop
        '''
        self.features_to_drop = ["sig_id", "cp_type"]

    def gen_train_data(self, data_all, loaded_train, loaded_targets):
        '''
        Generate training dataset.

        Args:
          data_all (dataframe): input data of features for train and test data
          loaded_train (dataframe): input data of features for train data
          loaded_targets (dataframe): input data of known responses (binary) from MoA targets for train data

        Returns:
          train_df (dataframe): training inputs with dimensions
            [n_observations,n_features]
          test_df (dataframe): test inputs with dimensions
            [n_observations,n_features]
          X_test (arr): test inputs with dimensions
            [n_observations,n_features]
        '''
        data_all.drop(self.features_to_drop, axis = 1, inplace = True)
        try:
            loaded_targets.drop("sig_id", axis = 1, inplace = True)
        except:
            pass

        train_df = data_all[: loaded_train.shape[0]]
        train_df.reset_index(drop = True, inplace = True)

        test_df = data_all[train_df.shape[0]: ]
        test_df.reset_index(drop = True, inplace = True)
        X_test = test_df.values

        return train_df, test_df, X_test
