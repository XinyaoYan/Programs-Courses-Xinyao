# -*- coding: UTF-8 -*-
# 为方便测试，请统一使用 numpy、pandas、sklearn 三种包，如果实在有特殊需求，请单独跟助教沟通
import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
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

        self.y_train = df_train['label'].values

        data_preprocessing = SimpleImputer(strategy="mean")
        self.X_train = data_preprocessing.fit_transform(df_train.iloc[:, 0:83])
        self.X_test = data_preprocessing.transform(df_test.iloc[:, 0:83])
        
        # logistic regression -- best = 0.608696
        # self.classification_model = LogisticRegressionCV(multi_class='multinomial',max_iter=3000)
        
        # decision tree -- best = 0.684564
        self.classification_model = DTC(criterion="entropy", max_depth=6)

        # random forest -- best = 0.619048
        # self.classification_model = RFC(n_estimators=120, random_state=0)

        # KNN -- best = 0.531469
        # self.classification_model = KNC(n_neighbors=5)

        # naive bayes -- best = 0.526946
        # self.classification_model = GaussianNB()
        # self.classification_model = MultinomialNB()
        # self.classification_model = BernoulliNB()

        self.df_predict = pd.DataFrame(index=df_test.index)

    # 模型训练，输出训练集f1_score
    def train(self):
        X_train, X_valid, y_train, y_valid = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=100) # 选择20%为测试集
        self.classification_model.fit(X_train, y_train)
        y_valid_pred = self.classification_model.predict(X_valid)
        return f1_score(y_valid, y_valid_pred)

    # 模型测试，输出测试集预测结果，要求此结果为DataFrame格式，可以通过to_csv方法保存为Kaggle的提交文件
    def predict(self):
        y_test_pred = self.classification_model.predict(self.X_test)
        self.df_predict['Predicted'] = y_test_pred
        return self.df_predict


# 以下部分请勿改动！
if __name__ == '__main__':
    # 解析输入参数。在终端执行以下语句即可运行此代码： python f_model.py --train_path "f_train.csv" --test_path "f_test.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="f_train.csv", help="path to train dataset")
    parser.add_argument("--test_path", type=str, default="f_test.csv", help="path to test dataset")
    opt = parser.parse_args()

    model = Model(opt.train_path, opt.test_path)
    print('训练集维度:{}\n测试集维度:{}'.format(model.X_train.shape, model.X_test.shape))
    f1_score_train = model.train()
    print('f1_score_train={:.6f}'.format(f1_score_train))
    f_predict = model.predict()
    f_predict.to_csv('f_predict.csv')
