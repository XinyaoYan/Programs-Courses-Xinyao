# -*- coding: UTF-8 -*-
# 为方便测试，请统一使用 numpy、pandas、sklearn 三种包，如果实在有特殊需求，请单独跟助教沟通
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression   #引入多元线性回归算法模块进行相应的训练
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#导入warnings包，利用过滤器来实现忽略警告语句。
import warnings
warnings.filterwarnings('ignore')

import argparse

# 设定随机数种子，保证代码结果可复现
np.random.seed(1024)


class Model:
    """
    要求:
        1. 需要有__init__、train、predict三个方法，且方法的参数应与此样例相同
        2. 需要有self.X_train、self.y_train、self.X_test三个实例变量，请注意大小写
        3. 如果划分出验证集，请将实例变量命名为self.X_valid、self.y_valid
    """
    # 模型初始化，数据预处理，仅为示例
    def __init__(self, train_path, test_path):
        df_train = pd.read_csv(train_path, encoding='gbk', index_col='id')
        df_test = pd.read_csv(test_path, encoding='gbk', index_col='id')
        feature_name = df_train.columns

        # df_train : 将 男/女 映射为 0/1
        class_mapping = {"男": 0, "女": 1}
        df_train["性别"] = df_train["性别"].map(class_mapping)

        # df_train : 将日期转化为天数
        df_train["体检日期"] = pd.to_datetime(df_train["体检日期"], format="%d/%m/%Y")
        df_train["体检日期"] = pd.to_datetime("24/12/2021", format="%d/%m/%Y") - df_train["体检日期"]
        df_train["体检日期"] = df_train["体检日期"].dt.days

        # df_train : 用均值填充nan值
        data_preprocessing = SimpleImputer(strategy='mean')
        for i in range(41):
            df_train[feature_name[i]] = data_preprocessing.fit_transform(df_train[feature_name[i]].values.reshape(-1, 1))
                
        # df_test : 将 男/女 映射为 0/1
        class_mapping = {"男": 0, "女": 1}
        df_test["性别"] = df_test["性别"].map(class_mapping)

        # df_test : 将日期转化为天数
        df_test["体检日期"] = pd.to_datetime(df_test["体检日期"], format="%d/%m/%Y")
        df_test["体检日期"] = pd.to_datetime("24/12/2021", format="%d/%m/%Y") - df_test["体检日期"]
        df_test["体检日期"] = df_test["体检日期"].dt.days

        # df_test : 用均值填充nan值
        data_preprocessing = SimpleImputer(strategy='mean')
        for i in range(40):
            df_test[feature_name[i]] = data_preprocessing.fit_transform(df_test[feature_name[i]].values.reshape(-1, 1))
                
        self.X_train = df_train.loc[:, feature_name[0:40]].values
        self.y_train = df_train.loc[:, '血糖'].values
        self.X_test = df_test.loc[:, feature_name[0:40]].values

        self.regression_model = LinearRegression()
        self.df_predict = pd.DataFrame(index=df_test.index)

    # 模型训练，输出训练集MSE
    def train(self):
        X_train, X_valid, y_train, y_valid = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=100)#选择20%为测试集
        self.regression_model.fit(X_train, y_train)
        print("最佳拟合线:截距\n", self.regression_model.coef_)     # 输出多元线性回归的各项系数
        print("回归系数:", self.regression_model.intercept_)        # 输出多元线性回归的常数项的值
        y_valid_pred = self.regression_model.predict(X_valid)
        print('r2_score={:.6f}'.format(r2_score(y_valid, y_valid_pred)))
        return mean_squared_error(y_valid, y_valid_pred)

    # 模型测试，输出测试集预测结果，要求此结果为DataFrame格式，可以通过to_csv方法保存为Kaggle的提交文件
    def predict(self):
        y_test_pred = self.regression_model.predict(self.X_test)
        self.df_predict['Predicted'] = y_test_pred
        return self.df_predict


# 以下部分请勿改动！
if __name__ == '__main__':
    # 解析输入参数。在终端执行以下语句即可运行此代码： python d_model.py --train_path "d_train.csv" --test_path "d_test.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="d_train.csv", help="path to train dataset")
    parser.add_argument("--test_path", type=str, default="d_test.csv", help="path to test dataset")
    opt = parser.parse_args()

    model = Model(opt.train_path, opt.test_path)
    print('训练集维度:{}\n测试集维度:{}'.format(model.X_train.shape, model.X_test.shape))
    MSE_train = model.train()
    print('MSE_train={:.6f}'.format(MSE_train))
    d_predict = model.predict()
    d_predict.to_csv('d_predict.csv')
