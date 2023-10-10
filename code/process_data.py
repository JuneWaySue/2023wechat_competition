import os,shutil,pickle,sys
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

from utils import seed_everything
seed_everything(2023)

def processing_data(now_path,user_id):
    try:
        log, trace, metric = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        log_file = f"{now_path}/log/{user_id}_log.csv"
        trace_file = f"{now_path}/trace/{user_id}_trace.csv"
        metric_file = f"{now_path}/metric/{user_id}_metric.csv"
        if os.path.exists(log_file):
            log = pd.read_csv(log_file).sort_values(by=['timestamp']).reset_index(drop=True)
        
        if os.path.exists(trace_file):
            trace = pd.read_csv(trace_file).sort_values(by=['timestamp']).reset_index(drop=True)
            
        if os.path.exists(metric_file):
            metric = pd.read_csv(metric_file).sort_values(by=['timestamp']).reset_index(drop=True)
        
        feats = {"id" : user_id}
        # log
        if len(log) > 0:
            for col in ['service']:
                if 'testing' not in now_path:
                    train_data_map[col] |= set(log[col].fillna('').astype(str).unique())
                else:
                    test_data_map[col] |= set(log[col].fillna('').astype(str).unique())
            
        # trace
        if len(trace) > 0:
            for col in ['service_name','endpoint_name','span_id','parent_id','host_ip']:
                if col != 'host_ip':
                    continue
                if 'testing' not in now_path:
                    if col == 'endpoint_name':
                        train_data_map[col] |= set(trace[trace[col].fillna('').astype(str).str.contains('GET|POST|Mysql',regex=True)][col].unique())
                    else:
                        train_data_map[col] |= set(trace[col].fillna('').astype(str).unique())
                else:
                    if col == 'endpoint_name':
                        test_data_map[col] |= set(trace[trace[col].fillna('').astype(str).str.contains('GET|POST|Mysql',regex=True)][col].unique())
                    else:
                        test_data_map[col] |= set(trace[col].fillna('').astype(str).unique())

            for i in trace['host_ip'].unique():
                for j in trace[trace['host_ip']==i]['service_name'].unique():
                    if 'testing' not in now_path:
                        train_data_map1[i].add(j)
                    else:
                        test_data_map1[i].add(j)

            for i in trace['host_ip'].unique():
                for j in trace[trace['host_ip']==i]['endpoint_name'].unique():
                    if 'GET' not in j and 'POST' not in j and 'Mysql' not in j:
                        continue
                    if 'testing' not in now_path:
                        train_data_map2[i].add(j)
                    else:
                        test_data_map2[i].add(j)
            
            for i in trace['service_name'].unique():
                for j in trace[trace['service_name']==i]['endpoint_name'].unique():
                    if 'GET' not in j and 'POST' not in j and 'Mysql' not in j:
                        continue
                    if 'testing' not in now_path:
                        train_data_map3[i].add(j)
                    else:
                        test_data_map3[i].add(j)
            
        # metric
        if len(metric) > 0:
            for col in ['tags']:
                if 'testing' not in now_path:
                    train_data_map[col] |= set(metric[col].fillna('').astype(str).unique())
                else:
                    test_data_map[col] |= set(metric[col].fillna('').astype(str).unique())
    
    except Exception as e:
        print(f'{user_id=}  {str(e)}')
        raise

    return feats

if os.path.exists('../other_dir/train_data_map.pkl') and os.path.exists('../other_dir/test_data_map.pkl'):
    print('train_data_map.pkl和test_data_map.pkl文件已存在，不用再次生成。')
    sys.exit()

train_data_map=defaultdict(set)
test_data_map=defaultdict(set)
train_data_map1=defaultdict(set)
test_data_map1=defaultdict(set)
train_data_map2=defaultdict(set)
test_data_map2=defaultdict(set)
train_data_map3=defaultdict(set)
test_data_map3=defaultdict(set)

save_path='../data/contest_data'
os.makedirs(save_path,exist_ok=True)
feature_list=[]
label_list=[]
for what in list(range(5)) + ['testing']:
    now_path=f'{save_path}/{what}'
    # if not os.path.exists(now_path):
    #     with py7zr.SevenZipFile(f'data_{what}.7z', mode='r') as archive:
    #         archive.extractall(path=save_path)

    user_ids = set([i.split("_")[0] for i in os.listdir(f"{now_path}/metric/")]) |\
            set([i.split("_")[0] for i in os.listdir(f"{now_path}/log/")]) |\
            set([i.split("_")[0] for i in os.listdir(f"{now_path}/trace/")])
    user_ids = list(user_ids)

    feature = pd.DataFrame(Parallel(n_jobs=os.cpu_count(), backend="threading")(delayed(processing_data)(now_path,user_id) for user_id in tqdm(user_ids,desc=f'{what}')))
    feature_list.append(feature)
    if what != 'testing':
        label = pd.read_csv(f'{now_path}/training_label_{what}.csv')
        label_list.append(label)

    # try:
    #     shutil.rmtree(now_path)
    # except:
    #     pass

os.makedirs('../other_dir',exist_ok=True)
label=pd.concat(label_list,ignore_index=True)
label.to_csv('../other_dir/label.csv',index=False)

train_data_map['host_ip_to_service_name']=dict(train_data_map1)
test_data_map['host_ip_to_service_name']=dict(test_data_map1)

train_data_map['host_ip_to_endpoint_name']=dict(train_data_map2)
test_data_map['host_ip_to_endpoint_name']=dict(test_data_map2)

train_data_map['service_name_to_endpoint_name']=dict(train_data_map3)
test_data_map['service_name_to_endpoint_name']=dict(test_data_map3)

with open('../other_dir/train_data_map.pkl','wb') as f:
    pickle.dump(train_data_map,f)
with open('../other_dir/test_data_map.pkl','wb') as f:
    pickle.dump(test_data_map,f)