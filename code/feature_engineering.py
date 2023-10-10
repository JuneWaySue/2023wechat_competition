import os,shutil,datetime,pickle,sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

from utils import seed_everything
seed_everything(2023)

def processing_feature(now_path,user_id):
    try:
        log, trace, metric = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        log_file = f"{now_path}/log/{user_id}_log.csv"
        trace_file = f"{now_path}/trace/{user_id}_trace.csv"
        metric_file = f"{now_path}/metric/{user_id}_metric.csv"
        if os.path.exists(log_file):
            log = pd.read_csv(log_file).sort_values(by=['timestamp'])
            log_start_num = log.shape[0]
            col = 'service'
            log = log[log[col].isin(train_data_map[col]&test_data_map[col])].reset_index(drop=True)
        
        if os.path.exists(trace_file):
            trace = pd.read_csv(trace_file).sort_values(by=['timestamp'])
            trace_start_num = trace.shape[0]
            col = 'service_name'
            trace = trace[trace[col].isin(train_data_map[col]&test_data_map[col])]
            col = 'host_ip'
            trace = trace[trace[col].isin(train_data_map[col]&test_data_map[col])].reset_index(drop=True)
            
        if os.path.exists(metric_file):
            metric = [1]
            # metric = pd.read_csv(metric_file).sort_values(by=['timestamp'])
            # col = 'tags'
            # metric = metric[metric[col].isin(train_data_map[col]&test_data_map[col])].reset_index(drop=True)
        
        feats = {"id" : user_id}
        # log
        if len(log) > 0:
            feats['log_sub_num'] = len(log) - log_start_num
            feats['log_length'] = len(log)
            feats['log_file_size'] = os.stat(log_file).st_size

            for col in ['message','service']:
                feats[f'log_{col}_nunique'] = log[col].fillna('').astype(str).nunique()
            
            text_list=['DEBUG','INFO','WARNING','ERROR','None']
            log['extract_name']='None'
            for text in text_list:
                if text == 'None':
                    log[text] = ~(log['message'].astype(str).str.contains('|'.join(text_list[:-1]),na=False))
                else:
                    log.loc[log['message'].str.contains(text),'extract_name']=text
                    log[text]=log['message'].astype(str).str.contains(text,na=False)

                for stats_func in ['sum', 'mean', 'std', 'skew', 'kurt']:
                    feats[f'log_message_{text}_{stats_func}'] = log[text].apply(stats_func)

            log['extract_name_timestamp_diff1'] = log.groupby('extract_name')['timestamp'].diff(1)
            log['service_timestamp_diff1'] = log.groupby('service')['timestamp'].diff(1)
            log['message_len'] = log['message'].fillna("").astype(str).str.len()
            for col_name in ['service','extract_name']:
                for i,tmp in log.groupby(col_name):
                    feats[f'log_{col_name}_{i}_num']=len(tmp)
                    for stats_func in ['mean', 'std', 'ptp', 'skew', 'kurt']:
                        feats[f'log_{col_name}_{i}_timestamp_diff1_{stats_func}']=tmp[f'{col_name}_timestamp_diff1'].dropna().apply(stats_func) if len(tmp) > 1 else np.nan
                        feats[f'log_{col_name}_{i}_message_len_{stats_func}']=tmp['message_len'].apply(stats_func)
                    if col_name == 'extract_name':
                        continue
                    for text in text_list:
                        for stats_func in ['sum', 'mean', 'std', 'skew', 'kurt']:
                            feats[f'log_{col_name}_{i}_{text}_{stats_func}'] = tmp[text].apply(stats_func)

            log['service&extract_timestamp_diff1'] = log.groupby(['service','extract_name'])['timestamp'].diff(1)
            for g_col,(n1,n2) in zip(
                        [['service','extract_name']],
                        [['service','extract']]
                    ):
                for (col1,col2),tmp in log.groupby(g_col):
                    feats[f'log_{n1}&{n2}_{col1}_{col2}_num']=len(tmp)
                    for stats_func in ['mean', 'std', 'ptp', 'skew', 'kurt']:
                        feats[f'log_{n1}&{n2}_{col1}_{col2}_timestamp_diff1_{stats_func}']=tmp[f'{n1}&{n2}_timestamp_diff1'].dropna().apply(stats_func) if len(tmp) > 1 else np.nan
                        feats[f'log_{n1}&{n2}_{col1}_{col2}_message_len_{stats_func}']=tmp['message_len'].apply(stats_func)
            
        # trace
        if len(trace) > 0:
            feats['trace_sub_num'] = len(trace) - trace_start_num
            feats['trace_length'] = len(trace)
            feats['trace_file_size'] = os.stat(trace_file).st_size

            for stats_func in ['mean','std']:
                feats[f"trace_status_code_{stats_func}"] = trace['status_code'].apply(stats_func)

            for stats_func in ['nunique']:
                for i in ['host_ip', 'service_name', 'endpoint_name', 'trace_id', 'span_id', 'start_time', 'end_time','status_code']:
                    feats[f"trace_{i}_{stats_func}"] = trace[i].apply(stats_func)

            text_list=['GET','POST','DELETE','Mysql','None']
            trace['extract_name']='None'
            for text in text_list:
                if text == 'None':
                    trace[text] = ~(trace['endpoint_name'].astype(str).str.contains('|'.join(text_list[:-1]),na=False))
                else:
                    trace.loc[trace['endpoint_name'].str.contains(text),'extract_name']=text
                    trace[text]=trace['endpoint_name'].astype(str).str.contains(text,na=False)

                for stats_func in ['sum', 'mean', 'std', 'skew', 'kurt']:
                    feats[f'trace_endpoint_name_{text}_{stats_func}'] = trace[text].apply(stats_func)

            trace['extract_name_timestamp_diff1'] = trace.groupby('extract_name')['timestamp'].diff(1)
            trace['status_code_timestamp_diff1'] = trace.groupby('status_code')['timestamp'].diff(1)
            trace['host_ip_timestamp_diff1'] = trace.groupby('host_ip')['timestamp'].diff(1)
            trace['service_name_timestamp_diff1'] = trace.groupby('service_name')['timestamp'].diff(1)
            trace['endpoint_name_timestamp_diff1'] = trace.groupby('endpoint_name')['timestamp'].diff(1)
            trace['time_sub']=trace['end_time'].clip(lower=0)-trace['start_time'].clip(lower=0)
            for col_name in ['service_name','host_ip','extract_name','status_code','endpoint_name']:
                for i,tmp in trace.groupby(col_name):
                    if (col_name == 'endpoint_name') and (i not in train_data_map[col_name] & test_data_map[col_name]):
                        continue
                    
                    feats[f'trace_{col_name}_{i}_num']=len(tmp)
                    for stats_func in ['mean', 'std', 'ptp', 'skew', 'kurt']:
                        feats[f'trace_{col_name}_{i}_timestamp_diff1_{stats_func}']=tmp[f'{col_name}_timestamp_diff1'].dropna().apply(stats_func) if len(tmp) > 1 else np.nan
                        feats[f'trace_{col_name}_{i}_time_sub_{stats_func}']=tmp['time_sub'].apply(stats_func)
                    
                    if col_name in ['extract_name','endpoint_name']:
                        continue
                    for text in text_list:
                        for stats_func in ['sum', 'mean', 'std', 'skew', 'kurt']:
                            feats[f'trace_{col_name}_{i}_{text}_{stats_func}'] = tmp[text].apply(stats_func)

            trace['service&extract_timestamp_diff1'] = trace.groupby(['service_name','extract_name'])['timestamp'].diff(1)
            trace['service&ip_timestamp_diff1'] = trace.groupby(['service_name','host_ip'])['timestamp'].diff(1)
            trace['ip&extract_timestamp_diff1'] = trace.groupby(['host_ip','extract_name'])['timestamp'].diff(1)
            trace['endpoint&ip_timestamp_diff1'] = trace.groupby(['endpoint_name','host_ip'])['timestamp'].diff(1)
            trace['endpoint&service_timestamp_diff1'] = trace.groupby(['endpoint_name','service_name'])['timestamp'].diff(1)
            for g_col,(n1,n2) in zip(
                        [['service_name','extract_name'],['service_name','host_ip'],['host_ip','extract_name'],['endpoint_name','host_ip'],['endpoint_name','service_name']],
                        [['service','extract'],          ['service','ip'],          ['ip','extract'],          ['endpoint','ip'],          ['endpoint','service']]
                    ):
                for (col1,col2),tmp in trace.groupby(g_col):
                    if (n1 == 'service' and n2 == 'ip') and (col1 not in train_data_map['host_ip_to_service_name'][col2] & test_data_map['host_ip_to_service_name'][col2]):
                        continue
                    if (n1 == 'endpoint' and n2 == 'ip') and (col1 not in train_data_map['host_ip_to_endpoint_name'][col2] & test_data_map['host_ip_to_endpoint_name'][col2]):
                        continue
                    if (n1 == 'endpoint' and n2 == 'service') and (col1 not in train_data_map['service_name_to_endpoint_name'][col2] & test_data_map['service_name_to_endpoint_name'][col2]):
                        continue
                    feats[f'trace_{n1}&{n2}_{col1}_{col2}_num']=len(tmp)
                    for stats_func in ['mean', 'std', 'ptp', 'skew', 'kurt']:
                        feats[f'trace_{n1}&{n2}_{col1}_{col2}_time_sub_{stats_func}']=tmp['time_sub'].apply(stats_func)
                        feats[f'trace_{n1}&{n2}_{col1}_{col2}_timestamp_diff1_{stats_func}']=tmp[f'{n1}&{n2}_timestamp_diff1'].dropna().apply(stats_func) if len(tmp) > 1 else np.nan

        # metric
        if len(metric) > 0:
            # use_cols=[
            #     'metric_name','service_name',
            #     'container','instance','job','kubernetes_io_hostname',
            #     'namespace','cpu','interface','kubernetes_pod_name','mode','minor','broadcast','duplex',
            #     'operstate'
            # ]
            # def process(x):
            #     item=eval(x['tags'])
            #     output=[]
            #     for col in use_cols:
            #         tmp=item[col] if col in item else '无'
            #         output.append(tmp)
            #     return output
            # metric[use_cols]=metric.apply(process, result_type='expand', axis=1)

            feats['metric_has_file'] = 1
            # feats['metric_length'] = len(metric)
            # feats['metric_file_size'] = os.stat(metric_file).st_size

            # for stats_func in ['mean', 'std', 'ptp', 'skew', 'kurt']:
            #     feats[f'metric_value_{stats_func}'] = metric['value'].apply(stats_func)

            # for col in ['timestamp']+use_cols:
            #     for stats_func in ['mean', 'std', 'ptp', 'skew', 'kurt']:
            #         feats[f'metric_{col}_value_sum_{stats_func}'] = metric[[col,'value']].groupby(col)['value'].sum().dropna().apply(stats_func)
            #         feats[f'metric_{col}_value_mean_{stats_func}'] = metric[[col,'value']].groupby(col)['value'].mean().dropna().apply(stats_func)
            #         tmp=metric[[col,'value']].groupby(col)['value'].std().dropna()
            #         feats[f'metric_{col}_value_std_{stats_func}'] = tmp.apply(stats_func) if len(tmp) > 0 else np.nan
            #         tmp=metric[[col,'value']].groupby(col)['value'].skew().dropna()
            #         feats[f'metric_{col}_value_skew_{stats_func}'] = tmp.apply(stats_func) if len(tmp) > 0 else np.nan

            # for col in ['timestamp']+use_cols:
            #     for i,tmp in metric[[col,'value']].groupby(col):
            #         if (col == 'kubernetes_pod_name' and i == 'node-exporter-52ljc'):
            #             continue
            #         if (col == 'container' and i == 'init-mysql'):
            #             continue
            #         feats[f'metric_{col}_{i}_num']=len(tmp)
            #         for stats_func in ['mean', 'std', 'ptp', 'skew', 'kurt']:
            #             feats[f'metric_{col}_{i}_value_{stats_func}']=tmp['value'].apply(stats_func)
        else:
            feats['metric_has_file'] = 0
    
    except Exception as e:
        print(f'{user_id=}  {str(e)}')
        raise

    return feats

if os.path.exists(f'../other_dir/feature.csv'):
    print('特征矩阵feature.csv文件已存在，不用再次生成。')
    sys.exit()

with open('../other_dir/train_data_map.pkl','rb') as f:
    train_data_map=pickle.load(f)
with open('../other_dir/test_data_map.pkl','rb') as f:
    test_data_map=pickle.load(f)

save_path='../data/contest_data'
os.makedirs(save_path,exist_ok=True)
feature_list=[]
for what in list(range(5)) + ['testing']:
    now_path=f'{save_path}/{what}'
    # if not os.path.exists(now_path):
    #     with py7zr.SevenZipFile(f'data_{what}.7z', mode='r') as archive:
    #         archive.extractall(path=save_path)

    user_ids = set([i.split("_")[0] for i in os.listdir(f"{now_path}/metric/")]) |\
            set([i.split("_")[0] for i in os.listdir(f"{now_path}/log/")]) |\
            set([i.split("_")[0] for i in os.listdir(f"{now_path}/trace/")])
    user_ids = list(user_ids)

    feature = pd.DataFrame(Parallel(n_jobs=os.cpu_count(), backend="threading")(delayed(processing_feature)(now_path,user_id) for user_id in tqdm(user_ids,desc=f'{what}')))
    feature_list.append(feature)

    # try:
    #     shutil.rmtree(now_path)
    # except:
    #     pass

feature=pd.concat(feature_list,ignore_index=True)

# 统一顺序，方便复现。
feature=feature.sort_values(by='id')
feature=feature[feature.columns.sort_values()]
feature=feature.sample(frac=1,random_state=2023).reset_index(drop=True)
col=feature.columns.tolist()
np.random.shuffle(col)
feature=feature[col]
print('feature.shape',feature.shape)

os.makedirs('../other_dir',exist_ok=True)
feature.to_csv(f'../other_dir/feature.csv',index=False)