# # 下载python3.8.8版本
# wget https://www.python.org/ftp/python/3.8.8/Python-3.8.8.tgz

# # 解压
# tar xzf Python-3.8.8.tgz

# # 安装python需要的库
# yum -y install gcc gcc-c++ openssl-devel bzip2-devel expat-devel gdbm-devel readline-devel sqlite-devel libffi-devel

# # 进入目录
# cd Python-3.8.8/

# # 进行配置，把安装的python​3.8.8的目录放到/usr/local/python3.8里面
# ./configure --prefix=/usr/local/python3

# # 完成后进行编译安装
# make -j2 && make install -j2

# # 退回上一级
# cd ..

# 安装依赖
pip install -r requirements.txt