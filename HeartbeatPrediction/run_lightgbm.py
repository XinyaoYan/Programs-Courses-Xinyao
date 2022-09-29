import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')

# 导入数据
path = "./data/"
train_data = pd.read_csv(path + 'train.csv')
test_data = pd.read_csv(path + 'testA.csv')
print('Train data shape:', train_data.shape)
print('TestA data shape:', test_data.shape)

# 构建评分特征
def abs_sum(y_pre,y_tru):
    # y_pre为预测概率矩阵
    # y_tru为真实类别矩阵
    y_pre = np.array(y_pre)
    y_tru = np.array(y_tru)
    loss = sum(sum(abs(y_pre-y_tru)))
    return loss

# 数据预处理
def reduce_mem_usage(df):
    # 显示数据框所有的内存情况
    start_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    # 循环每一列名：'id', 'heartbeat_signals', 'label'
    for col in df.columns:
        col_type = df[col].dtype # 参考每一列的数据类型
        
        # 判断：如果列类型不是对象的话则取出该列的最大值和最小值
        if col_type !=  object: 
            c_min = df[col].min()
            c_max = df[col].max()
            
            # 进一步判断是否为int属性
            if str(col_type)[:3] ==  'int':
                
                # 这里为了将数值小的转化为int8，节省空间，以下均是依次依次分段找到最为合适最省空间的格式
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

# 简单预处理
train_list = []   # 设置空列表

# 将ID以及心跳信号以及标签分布数字化
for items in train_data.values:
    train_list.append([items[0]]  +  [float(i) for i in items[1].split(',')]  +  [items[2]])

# 重新完成构建训练集
train = pd.DataFrame(np.array(train_list))
# 重新命名列名，其中将心跳信号名字以s开头进行命名
train.columns = ['id']  +  ['s_' + str(i) for i in range(len(train_list[0])-2)]  +  ['label']
# 节省空间
train = reduce_mem_usage(train)

# 对训练集同样处理
test_list = []
for items in test_data.values:
    test_list.append([items[0]]  +  [float(i) for i in items[1].split(',')])

test = pd.DataFrame(np.array(test_list))
test.columns = ['id']  +  ['s_' + str(i) for i in range(len(test_list[0])-1)]
test = reduce_mem_usage(test)

# 训练集和测试数据准备
x_train = train.drop(['id','label'], axis = 1)   # 去除掉id和因变量
y_train = train['label']                         # label作为因变量
x_test = test.drop(['id'], axis = 1)

# 训练模型,使用交叉验证
def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5                                    # 5折交叉验证
    seed = 2021                                  # 设置种子

    # 根据N折划分数据
    kf = KFold(n_splits = folds, shuffle = True, random_state = seed)   

    # 设置测试集，输出矩阵。每一组数据输出：[0,0,0,0]以概率值填入
    test = np.zeros((test_x.shape[0],4))

    # 交叉验证分数
    cv_scores = []
    onehot_encoder = OneHotEncoder(sparse = False)

    # 将训练集(K折)操作，i值代表第(i+1)折
    # 每一个K折都进行数据混乱————随机操作
    # train_index：用于训练的(K-1)的样本索引值
    # valid_index：剩下1折样本索引值，用于给出(训练误差)
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        # 打印第(i+1)个模型结果
        print('************************************ {} ************************************'.format(str(i + 1)))
        
        # 将训练集分为真正训练的数据(K-1折)，和训练集中的测试数据(1折)
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        print(trn_x, trn_y, val_x, val_y)
        if clf_name == "lgb":

            # 训练样本
            train_matrix = clf.Dataset(trn_x, label = trn_y)
            # 训练集中测试样本
            valid_matrix = clf.Dataset(val_x, label = val_y)
            
            # 参数设置
            params = {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': 4,
                'num_leaves': 60,
                'max_depth': 7,
                'min_data_in_leaf': 500,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'learning_rate': 0.1,
                'seed': seed,
                'nthread': 28,
                'n_jobs': 24,
                'verbose': -1,
            }

            # 模型
            model = clf.train(params, 
                      train_set = train_matrix, 
                      valid_sets = valid_matrix, 
                      num_boost_round = 2000, 
                      verbose_eval = 100, 
                      early_stopping_rounds = 200)

            val_pred = model.predict(val_x, num_iteration = model.best_iteration)
            test_pred = model.predict(test_x, num_iteration = model.best_iteration) 
        
        # 将列表转化为n行一列的矩阵，便于下一步转化为onehot
        val_y = np.array(val_y).reshape(-1, 1)  
        val_y = onehot_encoder.fit_transform(val_y)
        print('预测的概率矩阵为:')
        print(test_pred)

        # 将预测结果填入到test里面，这是一个(i个模型结果累加过程)
        test  +=  test_pred
        score = abs_sum(val_y, val_pred)
        cv_scores.append(score)
        print(cv_scores)
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    test = test/kf.n_splits

    return test
    
def lgb_model(x_train, y_train, x_test):
    lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_test

lgb_test = lgb_model(x_train, y_train, x_test)

temp = pd.DataFrame(lgb_test)

result = pd.read_csv('./data/sample_submit.csv')
result['label_0'] = temp[0]
result['label_1'] = temp[1]
result['label_2'] = temp[2]
result['label_3'] = temp[3]
result.to_csv('./data/submit.csv',index = False)