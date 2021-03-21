import torch
import os
import random
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import roc_auc_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
os.environ["PYTHONHASHSEED"] = str(42)

class Config(object):
    '''
    Load model parameters.

    Attributes:
      MAX_EPOCH (int): number of epochs
      NB_SPLITS (int): number of splits
      tabnet_params (dict): dictionary of TabNet model parameters
    '''
    def __init__(self):
        self.MAX_EPOCH = 200
        self.NB_SPLITS = 12
        self.tabnet_params = dict(
            n_d = 32,
            n_a = 32,
            n_steps = 1,
            gamma = 1.3,
            lambda_sparse = 0,
            optimizer_fn = optim.Adam,
            optimizer_params = dict(lr = 2e-2, weight_decay = 1e-5),
            mask_type = "entmax",
            scheduler_params = dict(mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9),
            scheduler_fn = ReduceLROnPlateau,
            seed = 42,
            verbose = 10)

class LogitsLogLoss(Metric):
    def __init__(self):
        '''
        Load parameters for custom loss function.

        Attributes:
          _name (str): name
          _maximize (bool): minimum solution
        '''
        self._name = "logits_ll"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        '''
        Load custom loss function.

        Args:
          y_true (arr): list of known values
          y_pred (arr): list of predicted values

        Returns:
          np.mean(-aux) (float): log loss value
        '''
        logits = 1 / (1 + np.exp(-y_pred))
        aux = (1 - y_true) * np.log(1 - logits + 1e-15) + y_true * np.log(logits + 1e-15)
        return np.mean(-aux)

class RunTabnet:
    def __init__(self, MAX_EPOCH, NB_SPLITS):
        '''
        Load number of epochs and splits for multi-label stratified k-fold cross validation.

        Args:
          MAX_EPOCH (int): number of epochs
          NB_SPLITS (int): number of splits

        Attributes:
          MAX_EPOCH (int): number of epochs
          NB_SPLITS (int): number of splits
        '''
        self.MAX_EPOCH =  MAX_EPOCH
        self.NB_SPLITS =  NB_SPLITS

    def run_model(self, train_df, targets, X_test, tabnet_params):
        '''
        Run model.

        Args:
          train_df (dataframe): training inputs with dimensions
            [n_observations,n_features]
          targets (dataframe): updated input data of known responses (binary) from MoA targets for train data
          X_test (arr): test inputs with dimensions
            [n_observations,n_features]
          tabnet_params (dict): dictionary of TabNet model parameters

        Returns:
          test_preds_all (arr): predicted outputs with dimensions
            [n_splits_kfold,n_observations,n_moa_targets]
        '''
        test_cv_preds = []
        oof_preds = []
        oof_targets = []
        scores = []

        mskf = MultilabelStratifiedKFold(n_splits = self.NB_SPLITS, random_state = 0, shuffle = True)

        for fold_nb, (train_idx, val_idx) in enumerate(mskf.split(train_df, targets)):
            print("FOLDS: ", fold_nb + 1)
            print('*' * 60)

            X_train, y_train = train_df.values[train_idx, :], targets.values[train_idx, :]
            X_val, y_val = train_df.values[val_idx, :], targets.values[val_idx, :]

            model = TabNetRegressor(**tabnet_params)

            model.fit(
                X_train = X_train,
                y_train = y_train,
                eval_set = [(X_val, y_val)],
                eval_name = ["val"],
                eval_metric = ["logits_ll"],
                max_epochs = self.MAX_EPOCH,
                patience = 20,
                batch_size = 1024,
                virtual_batch_size = 32,
                num_workers = 1,
                drop_last = False,
                loss_fn = F.binary_cross_entropy_with_logits)
            print('-' * 60)

            preds_val = model.predict(X_val)
            preds = 1 / (1 + np.exp(-preds_val))
            score = np.min(model.history["val_logits_ll"])

            oof_preds.append(preds_val)
            oof_targets.append(y_val)
            scores.append(score)

            preds_test = model.predict(X_test)
            test_cv_preds.append(1 / (1 + np.exp(-preds_test)))

        oof_preds_all = np.concatenate(oof_preds)
        oof_targets_all = np.concatenate(oof_targets)
        test_preds_all = np.stack(test_cv_preds)

        aucs = []
        for task_id in range(oof_preds_all.shape[1]):
            aucs.append(roc_auc_score(y_true = oof_targets_all[:, task_id],y_score = oof_preds_all[:, task_id]))

        print(f"Overall AUC: {np.mean(aucs)}")
        print(f"Average CV: {np.mean(scores)}")

        return test_preds_all

    def gen_csv(self, test_preds_all, test, submission):
        '''
        Generate the CSV file for predicted response from MoA targets.

        Args:
          test_preds_all (arr): predicted outputs with dimensions
            [n_splits_kfold,n_observations,n_moa_targets]
          test (dataframe): input data of features for test data
          submission(dataframe): input data of predicted responses from MoA targets for test data

        Returns:
          submission (dataframe): predicted response from MoA targets for test data
        '''
        all_feat = [col for col in submission.columns if col not in ["sig_id"]]

        sig_id = test[test["cp_type"] != "ctl_vehicle"].sig_id.reset_index(drop = True)
        tmp = pd.DataFrame(test_preds_all.mean(axis = 0), columns = all_feat)
        tmp["sig_id"] = sig_id

        submission = pd.merge(test[["sig_id"]], tmp, on = "sig_id", how = "left")
        submission.fillna(0, inplace = True)

        submission.to_csv("submission.csv", index = None)

        return submission
