import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from utils import seed_everything,baseline,get_train_data,sScore

class CFG:
    feature_path = '../other_dir/feature.csv'
    model_path = '../data/model_data'
    model_type = 'lgb_ovr'
    num_classes = 9
    n_splits = 5

    feature_pre_filter = False
    boosting_type = 'gbdt'
    objective = 'binary'
    metric = 'auc'
    seed = 2023
    learning_rate = 0.18
    colsample_bytree = 0.8
    reg_lambda = 1.5
    num_leaves = 31 # 默认
    max_depth = -1 # 默认
    n_jobs = -1
    verbose = -1

    def keys(self):
        # 当用dict()将CFG类转为dict时，指定变量返回
        return {'feature_pre_filter', 'boosting_type', 'objective', 'metric', 'seed', 'learning_rate', 'colsample_bytree', 'reg_lambda', 'num_leaves', 'max_depth', 'n_jobs', 'verbose'} & {i for i in CFG.__dict__.keys()}

    def __getitem__(self, item):
        v = getattr(self, item)
        v = v[0] if isinstance(v,tuple) and len(v) == 1 else v
        return v

args=CFG()
seed_everything(args.seed)

all_data,train_X,test_X,train_scaler_X,test_scaler_X,X,y,lb_encoder=get_train_data(args)
print(f"train_X shape = {train_X.shape}")
print(f"test_X shape = {test_X.shape}")
print(f"Feature = {train_X.columns.tolist()}")

ovr_preds,ovr_oof = baseline(args,train_scaler_X,test_scaler_X,y)

each_score = sScore(y, ovr_oof)
score_metric = pd.DataFrame(each_score, columns=['score'], index=list(lb_encoder.classes_))
s_auc = np.mean(score_metric['score'])
score_metric.loc["Weighted AVG.", "score"] = s_auc
print(score_metric)

# submit = pd.DataFrame(ovr_preds, columns=lb_encoder.classes_)
# submit.index = X[all_data['label'].isnull()].index
# submit.reset_index(inplace=True)
# submit = submit.melt(id_vars="id", value_vars=lb_encoder.classes_, value_name="score", var_name="source")
# submit.to_csv(f"../data/tmp_data/submit_{args.model_type}.csv", index=False)