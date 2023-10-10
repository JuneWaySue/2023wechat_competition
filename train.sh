# 进入code目录
cd code

# 预处理数据，提取训练集和测试集中，部分字段所有出现过的集合
python process_data.py

# 特征工程
python feature_engineering.py

# 训练ovr lgb模型一
python lgb_ovr.py

# 训练ovr 9个二分类模型二
python lgb_9.py