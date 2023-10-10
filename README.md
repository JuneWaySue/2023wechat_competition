- blog：<a href="https://blog.csdn.net/sinat_39629323/article/details/133701640" target="_blank">2023微信大数据挑战赛—参赛总结</a>
- wechat： **Python王者之路**

![Python王者之路](https://user-images.githubusercontent.com/45711125/135013611-4c5d58da-bdac-4034-a93b-8d1c66899b53.jpg)

# 代码说明
- `code/process_data.py` ：数据预处理，主要是提取训练集和测试集中，部分字段所有出现过的集合，然后在特征提取时只针对训练集和测试集中共有的部分做特征，我认为这样能在一定程度上缓解所构建的特征中训练集和测试集分布不一致的问题，最后会输出；`train_data_map.pkl`和`test_data_map.pkl`两个文件；
- `code/feature_engineering.py` ：特征工程，即对每个id（包括测试集）提取特征，然后输出`feature.csv`文件，用于后续的模型训练；
- `code/utils.py` ：一些自定义函数和模型训练的函数等等；
- `code/lgb_ovr.py` ：模型一的训练入口文件，运行之后会生成每一折的模型文件；
- `code/lgb_9.py` ：模型二的训练入口文件，运行之后会生成每个标签每一折的模型文件；
- `code/prediction.py` ：得到测试集最终的预测结果文件入口文件，运行之后会在`data/submission/`下生成`result.csv`文件；

# 上传的模型文件的说明（运行test.sh时会读取这里的模型进行预测）
`data/model_data/`目录下的所有文件是模型一和模型二每一折的模型文件，可直接用于测试集的预测，对结果进行复现。

# 其他上传文件的说明（这些都是为了能让复现更好地完成的，如果要严格复现所有过程，那么要先把下面的文件夹下的文件都删除）
- `other_dir/feature.csv` ：包含了训练集和测试集的特征矩阵文件，可直接用于预测测试集，方便复现结果；
- `other_dir/train_data_map.pkl` ：提取了训练集中，部分字段所有出现过的集合；
- `other_dir/test_data_map.pkl` ：提取了测试集中，部分字段所有出现过的集合；
- `other_dir/scaler.pkl` ：数据标准化`StandardScaler`对象序列化文件；
- `other_dir/lb_encoder.pkl` ：各个标签编码后`LabelEncoder`对象序列化文件；
- `other_dir/label.csv` ：训练集中各个id所对应的标签文件；

## 环境配置（必选）
python版本是3.8.8

## 数据（必选）
没有使用外部公开数据

## 预训练模型（必选）
没有使用预训练模型

## 算法（必选）
- 模型一：OneVsRestClassifier包装LGBMClassifier，即baseline里所使用的方法
- 模型二：分别构建9个lightgbm二分类模型

### 整体思路介绍（必选）
对每个id的log、trace、metric文件提取特征，
然后模型一和模型二均采用5折交叉验证的方式进行训练，
最后将两个模型对测试集进行预测得到的两个结果，按0.65和0.35的权重进行加权融合，得到最终的预测结果。

## 训练流程（必选）
在train.sh中对每一步添加了注释

## 测试流程（必选）
在test.sh中对每一步添加了注释