import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier

# 读入数据
train = pd.read_csv("./train/train.csv")
test = pd.read_csv("./train/testA.csv")

# 数据预处理
train_x = np.array(train["heartbeat_signals"].str.split(",", expand=True)).astype("float32").reshape(-1,205,1)
train_y = np.array(train["label"].astype("int8"))
test_x = np.array(test['heartbeat_signals'].str.split(',', expand=True)).astype("float32").reshape(-1,205,1)

# 用 KNN 模型进行一次预测
knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit(train_x.reshape(-1,205), train_y)
y_preknn = tf.one_hot(knn.predict(test_x.reshape(-1,205)), depth=4)

# 搭建第一个 CNN 模型
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding='same', input_shape=(205, 1),  activation='relu'),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')
])
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

# 搭建第二个 CNN 模型
model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=16, kernel_size=9, padding='same', input_shape=(205, 1),  activation='relu'),
    tf.keras.layers.Conv1D(filters=32, kernel_size=6, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4,activation='softmax')
])
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

# 训练两个 CNN 模型
model1.fit(train_x, train_y, epochs=30, batch_size=200,validation_split=0.1)
model2.fit(train_x, train_y, epochs=30, batch_size=200,validation_split=0.1)

# 用训练好的模型进行预测
y_pre1 = model1.predict(test_x)
y_pre2 = model2.predict(test_x)

# 将三次预测的结果融合
y_pre = np.array(y_pre1 + y_pre2 + y_preknn)
res = tf.one_hot(y_pre.argmax(axis=1), depth=4).numpy()

# 存储最终的预测结果
submit = pd.DataFrame()
submit["id"] = test["id"]
submit["label_0"] = res[:, 0]
submit["label_1"] = res[:, 1]
submit["label_2"] = res[:, 2]
submit["label_3"] = res[:, 3]
submit.to_csv("./train/sample_submit.csv", index=None)

