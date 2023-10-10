import os,random,pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder,StandardScaler

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def gen_label(train):
    col = np.zeros((train.shape[0], 9))
    for i, label in enumerate(train['label'].values):
        col[i][label] = 1
        
    return col

def sScore(y_true, y_pred):
    score = []
    for i in range(y_true.shape[1]):
        score.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        
    return score

def get_train_data(args):
    feature = pd.read_csv(args.feature_path)

    filter_cols=['id', 'label']
    for i in feature.columns:
        if feature[i].isna().sum() / len(feature) > 0.95:
            filter_cols.append(i)
        elif feature[i].nunique() == 1:
            filter_cols.append(i)

    label = pd.read_csv('../other_dir/label.csv')

    if os.path.exists('../other_dir/lb_encoder.pkl'):
        with open('../other_dir/lb_encoder.pkl','rb') as f:
            lb_encoder = pickle.load(f)
    else:
        lb_encoder = LabelEncoder()

    label['label'] = lb_encoder.fit_transform(label['source'])
    all_data = feature.merge(label[['id', 'label']].groupby(['id'], as_index=False)['label'].agg(list), how='left', on=['id']).set_index("id")
    feature_name = [i for i in all_data.columns if i not in filter_cols]
    X = all_data[feature_name].replace([np.inf, -np.inf], 0).clip(-1e9, 1e9)
    for col in feature_name:
        fillna=-1 if X[col].isna().sum() > 0 and X[col].min() == 0 else 0
        X[col]=X[col].fillna(fillna)
        
    y = gen_label(all_data[all_data['label'].notnull()])
    
    if os.path.exists('../other_dir/scaler.pkl'):
        with open('../other_dir/scaler.pkl','rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = StandardScaler()

    train_X = X[all_data['label'].notnull()]
    test_X = X[all_data['label'].isnull()]

    scaler.fit(train_X)
    train_scaler_X = scaler.transform(train_X)
    test_scaler_X = scaler.transform(test_X)

    if not os.path.exists('../other_dir/scaler.pkl'):
        with open('../other_dir/scaler.pkl','wb') as f:
            pickle.dump(scaler,f)
    if not os.path.exists('../other_dir/lb_encoder.pkl'):
        with open('../other_dir/lb_encoder.pkl','wb') as f:
            pickle.dump(lb_encoder,f)

    return all_data,train_X,test_X,train_scaler_X,test_scaler_X,X,y,lb_encoder

def baseline(args,train_X,test_X,y):
    kf = MultilabelStratifiedKFold(n_splits=args.n_splits, random_state=args.seed, shuffle=True)

    ovr_oof = np.zeros((len(train_X), args.num_classes))
    ovr_preds = np.zeros((len(test_X), args.num_classes))

    for fold,(train_index, valid_index) in enumerate(kf.split(train_X, y)):
        X_train, X_valid = train_X[train_index], train_X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        clf = OneVsRestClassifier(lgb.LGBMClassifier(**dict(args)))
        clf.fit(X_train, y_train)
        joblib.dump(clf, f'{args.model_path}/{args.model_type}_fold{fold}.joblib')
        ovr_oof[valid_index] = clf.predict_proba(X_valid)
        ovr_preds += clf.predict_proba(test_X) / args.n_splits
        score = sScore(y_valid, ovr_oof[valid_index])
        print(f"fold{fold+1}： Score = {np.mean(score)}")

    return ovr_preds,ovr_oof

def tree_train_binary(args,feature_names,train_scaler_X,test_scaler_X,Y,label_i):
    score = np.zeros(5)
    oof = np.zeros(len(train_scaler_X))
    preds = np.zeros(len(test_scaler_X))
    feature_importance_df = pd.DataFrame()
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    y=deepcopy(Y)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_scaler_X, y)):
        tra = lgb.Dataset(train_scaler_X[train_idx],y[train_idx])
        val = lgb.Dataset(train_scaler_X[val_idx],y[val_idx])
        model = lgb.train(dict(args), tra, valid_sets=[val], num_boost_round=args.num_boost_round,
                        callbacks=[lgb.early_stopping(args.early_stopping), lgb.log_evaluation(args.num_boost_round)])

        joblib.dump(model, f'{args.model_path}/{args.model_type}_label{label_i}_fold{fold}.joblib')

        auc = model.best_score['valid_0']['auc']
        score[fold] = auc
        print(f'{fold=} {auc=}')

        oof[val_idx] = model.predict(train_scaler_X[val_idx], num_iteration=model.best_iteration)
        preds += model.predict(test_scaler_X, num_iteration=model.best_iteration) / args.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = feature_names
        fold_importance_df["importance"] = model.feature_importance()
        fold_importance_df["fold"] = fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df])

    print('auc mean',np.mean(score))
    
    return preds,oof,feature_importance_df
    