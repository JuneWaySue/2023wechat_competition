import joblib,os
import numpy as np
import pandas as pd

from utils import get_train_data

def get_pred_lgb_ovr(args,test_scaler_X):
    ovr_preds = np.zeros((len(test_scaler_X), args.num_classes))
    for fold in range(args.n_splits):
        clf = joblib.load(f'{args.model_path}/{args.model_type}_fold{fold}.joblib')
        ovr_preds += clf.predict_proba(test_scaler_X) / args.n_splits

    return ovr_preds

def get_pred_lgb_9(args,test_scaler_X):
    binary_preds=np.zeros((test_scaler_X.shape[0],args.num_classes))
    for label_i in range(9):
        preds = np.zeros(len(test_scaler_X))
        for fold in range(args.n_splits):
            model = joblib.load(f'{args.model_path}/{args.model_type}_label{label_i}_fold{fold}.joblib')
            preds += model.predict(test_scaler_X, num_iteration=model.best_iteration) / args.n_splits

        binary_preds[:,label_i]=preds

    return binary_preds

class CFG:
    feature_path = '../other_dir/feature.csv'
    model_path = '../data/model_data'
    model_type = ''
    n_splits = 5
    num_classes = 9

args=CFG()
all_data,train_X,test_X,train_scaler_X,test_scaler_X,X,y,lb_encoder=get_train_data(args)

args.model_type='lgb_ovr'
pred_lgb_ovr=get_pred_lgb_ovr(args,test_scaler_X)

args.model_type='lgb_9'
pred_lgb_9=get_pred_lgb_9(args,test_scaler_X)
pred=pred_lgb_ovr * 0.65 + pred_lgb_9 * 0.35

submit = pd.DataFrame(pred, columns=lb_encoder.classes_)
submit.index = X[all_data['label'].isnull()].index
submit.reset_index(inplace=True)
submit = submit.melt(id_vars="id", value_vars=lb_encoder.classes_, value_name="score", var_name="source")
submit.to_csv("../data/submission/result.csv", index=False)


