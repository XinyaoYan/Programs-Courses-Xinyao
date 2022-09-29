from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot
import glob
import os
import cv2
import  tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot
from skimage import io,transform
import tensorflow as tf
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


# 数据集地址
path='./flower_photos/'
# 模型保存地址
model_path='./model/model.ckpt'

# 将所有的图片resize成100*100
w = 100
h = 100
c = 3

#读取图片
def read_img(path):
    cate = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
data, label = read_img(path)

# 打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]

# 将所有数据分为训练集和验证集
ratio=0.8
s = np.int(num_example*ratio)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]

#-----------------构建网络----------------------

# ----------------创建placeholders-----------
# 使用None作为batch的大小，X的维度是[None, n_H0, n_W0, n_C0]， Y的维度是[None, n_y]
tf.compat.v1.disable_eager_execution()

x = tf.compat.v1.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_ = tf.compat.v1.placeholder(tf.int32,shape=[None,],name='y_')

# ----------------处理向前传播-----------------
# 封装一个创建卷积层的函数
def forward_propagation(input_tensor, weight, filters, strides):
    conv_weights = tf.compat.v1.get_variable("weight", weight, initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
    conv_biases = tf.compat.v1.get_variable("bias", [filters], initializer=tf.compat.v1.constant_initializer(0.0))
    conv = tf.nn.conv2d(input=input_tensor, filters=conv_weights, strides=strides, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    return conv_weights, conv_biases, conv, relu


def inference(input_tensor, train, regularizer):
    # -----------------------第1层----------------------------
    with tf.compat.v1.variable_scope('layer1-conv1'):
        # 大小为5×5，通道数为3（3个RGB通道），步长为1，核数为32
        conv1_weights, conv1_biases, conv1, relu1 = forward_propagation(input_tensor, [5, 5, 3, 32], 32, [1, 1, 1, 1])

    with tf.compat.v1.name_scope("layer2-pool1"):
        # 池化计算，池化边界为2，步长为2
        pool1 = tf.nn.max_pool2d(input=relu1, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    # -----------------------第2层----------------------------
    with tf.compat.v1.variable_scope("layer3-conv2"):
        # 大小为5×5，通道数为32，步长为1，核数为64
        conv2_weights, conv2_biases, conv2, relu2 = forward_propagation(pool1, [5, 5, 32, 64], 64, [1, 1, 1, 1])

    with tf.compat.v1.name_scope("layer4-pool2"):
        # 池化计算，池化边界为2，步长为2
        pool2 = tf.nn.max_pool2d(input=relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # -----------------------第3层----------------------------
    with tf.compat.v1.variable_scope("layer5-conv3"):
        # 大小为5×5，通道数为64，步长为1，核数为128
        conv3_weights, conv3_biases, conv3, relu3 = forward_propagation(pool2, [3, 3, 64, 128], 128, [1, 1, 1, 1])

    with tf.compat.v1.name_scope("layer6-pool3"):
        # 池化计算，池化边界为2，步长为2
        pool3 = tf.nn.max_pool2d(input=relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # -----------------------第4层----------------------------
    with tf.compat.v1.variable_scope("layer7-conv4"):
        # 大小为5×5，通道数为128，步长为1，核数为128
        conv4_weights, conv4_biases, conv4, relu4 = forward_propagation(pool3, [3, 3, 128, 128], 128, [1, 1, 1, 1])

    with tf.compat.v1.name_scope("layer8-pool4"):
        # 池化计算，池化边界为2，步长为2
        pool4 = tf.nn.max_pool2d(input=relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        nodes = 6*6*128
        reshaped = tf.reshape(pool4, [-1, nodes])

    # -----------------------第5层----------------------------
    with tf.compat.v1.variable_scope('layer9-fc1'):
        # 初始化全连接层的参数，隐含节点为1024个
        fc1_weights = tf.compat.v1.get_variable("weight", [nodes, 1024], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.compat.v1.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.compat.v1.get_variable("bias", [1024], initializer=tf.compat.v1.constant_initializer(0.1))
        # 使用relu函数作为激活函数
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        # 采用dropout层，减少过拟合和欠拟合的程度，保存模型最好的预测效率
        if train: fc1 = tf.nn.dropout(fc1, 0.7)

    with tf.compat.v1.variable_scope('layer10-fc2'):
        # 同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数
        fc2_weights = tf.compat.v1.get_variable("weight", [1024, 512], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.compat.v1.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.compat.v1.get_variable("bias", [512], initializer=tf.compat.v1.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.7)

    with tf.compat.v1.variable_scope('layer11-fc3'):
        fc3_weights = tf.compat.v1.get_variable("weight", [512, 6], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.compat.v1.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.compat.v1.get_variable("bias", [6], initializer=tf.compat.v1.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit
#---------------------------网络结束---------------------------

regularizer = tf.keras.regularizers.l2(0.5 * (0.0001))
logits = inference(x,False,regularizer)

b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval')

loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
train_op=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(input=logits,axis=1),tf.int32), y_)
acc= tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 训练和测试数据
n_epoch = 20
batch_size = 16
saver = tf.compat.v1.train.Saver()
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(n_epoch):
    start_time = time.time()

    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err
        train_acc += ac
        n_batch += 1
    print("--第" + str(epoch + 1) + "次训练:".ljust(10, " "), end="")
    print(str((np.sum(train_loss)) / n_batch).ljust(20, " "), (str(np.sum(train_acc) / n_batch)).ljust(20, " "),
          end="")

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err
        val_acc += ac
        n_batch += 1
    print(str((np.sum(val_loss) / n_batch)).ljust(20, " "), str((np.sum(val_acc) / n_batch)).ljust(20, " "))

saver.save(sess,model_path)
sess.close()

types = 0  # 所有花的种类数,也是标签的个数
# flower = {0: 'bee', 1: 'blackberry', 2: 'blanket', 3: 'bougainvillea', 4: 'bromelia', 5: 'foxglove'}
flower = {0:'none', 1: 'bee', 2: 'blackberry', 3: 'blanket', 4: 'bougainvillea', 5: 'bromelia', 6: 'foxglove'}

def get_test_img(path):
    imgs = []
    cv_data = []
    for im in os.listdir(path):
        im = path + im
        print('\rrecognizing :%s' % (im), end="")
        img = io.imread(im)
        img = transform.resize(img, (w, h))
        imgs.append(img)
        cv_data.append(cv2.imread(im))
    return np.asarray(imgs), cv_data

def recog():
    path = './TestImages/'
    model_path = "./model/"

    data, cv_datas = get_test_img(path)
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(model_path + 'model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        graph = tf.compat.v1.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: data}

        logits = graph.get_tensor_by_name("logits_eval:0")
        classification_result = sess.run(logits, feed_dict)

        # 打印出预测矩阵
        print(classification_result)
        # 打印出预测矩阵每一行最大值的索引
        print(tf.argmax(input=classification_result, axis=1).eval())
        # 根据索引通过字典对应花的分类
        output = tf.argmax(input=classification_result, axis=1).eval()
        for i in range(len(output)):
#             print("第", i + 1, flower[output[i]])
            print("第", i + 1, "朵花预测: label:", output[i], "\t", flower[output[i]])
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv_datas[i]
            cv2.putText(img, flower[output[i]], (20, 20), font, 1, (0, 0, 0), 1)
            # cv2.imshow(flower[output[i]],img)
            # cv2.waitKey(0)  # # 使图片停留用于观察，没有这一行代码，图片会在展示瞬间后消失
            # 下面改用pyplot绘图以让结果显示在一张图片中
            matplotlib.pyplot.subplot(2, 3, 1 + i)  # 使用subplot()构建子图，第一个参数5表示每列有5张图，第二个参数表示每行有6张图，1+i表示第(1+i张图)
            matplotlib.pyplot.axis('off')  # 关闭坐标轴("on"为显示坐标轴)
            matplotlib.pyplot.title(flower[output[i]])
            matplotlib.pyplot.imshow(img, cmap='gray_r')  # 使用imshow()函数画出训练集相应图片原始像素数据，参数cmap = gray表示黑底白字图像，这边使用"gray_r"表示白底黑字图
            matplotlib.pyplot.savefig('res.png')  # 保存图片
            matplotlib.pyplot.show()

recog()
