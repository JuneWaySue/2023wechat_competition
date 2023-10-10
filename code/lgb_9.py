import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from utils import seed_everything,get_train_data,tree_train_binary,sScore

class CFG:
    feature_path = '../other_dir/feature.csv'
    model_path = '../data/model_data'
    model_type = 'lgb_9'
    num_classes = 1
    n_splits = 5

    num_boost_round = 10000
    early_stopping = 50

    feature_pre_filter = False
    boosting_type = 'gbdt'
    objective = 'binary'
    metric = 'auc'
    seed = 2023
    n_jobs = -1
    verbose = -1

    def keys(self):
        # 当用dict()将CFG类转为dict时，指定变量返回
        return {'feature_pre_filter', 'boosting_type', 'objective', 'metric', 'seed', 'n_jobs', 'verbose'} & {i for i in CFG.__dict__.keys()}

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

feature_names=train_X.columns.tolist()
feature_importance_df=pd.DataFrame()
binary_oof=np.zeros_like(y)
binary_preds=np.zeros((test_scaler_X.shape[0],y.shape[1]))

for label_i in range(9):
    print(f'{label_i=}')

    preds,oof,label_i_feature_importance_df=tree_train_binary(args,feature_names,train_scaler_X,test_scaler_X,y[:,label_i],label_i)
    binary_oof[:,label_i]=oof
    binary_preds[:,label_i]=preds
    label_i_feature_importance_df=label_i_feature_importance_df[["Feature", "importance"]].groupby(["Feature"]).mean().reset_index()
    label_i_feature_importance_df['label_i']=label_i
    feature_importance_df=pd.concat([feature_importance_df,label_i_feature_importance_df])

    print('-'*88)
    break

each_score = sScore(y, binary_oof)
score_metric = pd.DataFrame(each_score, columns=['score'], index=list(lb_encoder.classes_))
s_auc = np.mean(score_metric['score'])
score_metric.loc["Weighted AVG.", "score"] = s_auc
print(score_metric)

# import matplotlib.pyplot as plt
# import seaborn as sns
# cols = (feature_importance_df
# 		.groupby("Feature")
# 		.mean()
# 		.sort_values(by="importance", ascending=False)[:100].index)

# best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

# plt.figure(figsize=(14,25))
# sns.barplot(x = "importance",
# 			y = "Feature",
# 			data = best_features.sort_values(by="importance", ascending=False))
# plt.title('LightGBM Features (avg over folds and over labels)')
# plt.tight_layout()

# submit = pd.DataFrame(binary_preds, columns=lb_encoder.classes_)
# submit.index = X[all_data['label'].isnull()].index
# submit.reset_index(inplace=True)
# submit = submit.melt(id_vars="id", value_vars=lb_encoder.classes_, value_name="score", var_name="source")
# submit.to_csv(f"../data/tmp_data/submit_{args.model_type}.csv", index=False)