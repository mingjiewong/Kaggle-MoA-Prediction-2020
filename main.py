from tabnet_moa.preprocessing import Load, ScaledPCA, Preprocess
from tabnet_moa.prediction import Config, LogitsLogLoss, RunTabnet
from gaussrank.gauss_rank_scaler import GaussRankScaler

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    train_filename = './datasets/lish-moa/train_features.csv'
    targets_filename = './datasets/lish-moa/train_targets_scored.csv'
    test_filename = './datasets/lish-moa/test_features.csv'
    submission_filename = './datasets/lish-moa/sample_submission.csv'

    load = Load(train_features=train_filename, train_targets_scored=targets_filename, test_features=test_filename, submission=submission_filename)
    loaded_train, loaded_test, loaded_targets = load.drop_ctl_vehicle()

    gauss_rank_pca = ScaledPCA(scaler=GaussRankScaler)
    scaled_data_all = gauss_rank_pca.run_scaling(loaded_train, loaded_test)
    concat_data_all = gauss_rank_pca.run_pca(scaled_data_all)
    encoded_data_all = gauss_rank_pca.one_hot_encoding(concat_data_all)

    preprocess = Preprocess()
    train_df, test_df, X_test = preprocess.gen_train_data(data_all=encoded_data_all, loaded_train=loaded_train, loaded_targets=loaded_targets)

    cfg = Config()
    run_tabnet = RunTabnet(MAX_EPOCH=cfg.MAX_EPOCH, NB_SPLITS=cfg.NB_SPLITS)
    test_preds_all = run_tabnet.run_model(train_df=train_df, targets=loaded_targets, X_test=X_test, tabnet_params=cfg.tabnet_params)

    submission_results = run_tabnet.gen_csv(test_preds_all=test_preds_all, test=loaded_test, submission=load.submission)
