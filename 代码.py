import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from matplotlib import style
import pandas as pd

#导入数据
data = pd.read_excel('澳元-人民币.xls',header = 0, sheetname='Sheet1')
data.head()

#获取时间及收盘价
time = data.iloc[:,0].tolist()
data = data.iloc[:,4].tolist()

#观察原数据基本特征
style.use('ggplot')
plt.figure(figsize=(16,9))
plt.rcParams['font.sans-serif'] = 'SimHei'   #设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False   #设置正常显示符号
plt.title('原始数据')
plt.plot(time,data)
plt.show()

#将原始数据进行缩放
def data_processing(raw_data,scale=True):
    if scale == True:
        return (raw_data-np.mean(raw_data))/np.std(raw_data)#标准化
    else:
        return (raw_data-np.min(raw_data))/(np.max(raw_data)-np.min(raw_data))#极差规格化

#设置基本参数    
HIDDEN_SIZE = 32  #隐层神经元个数
NUM_LAYERS = 1  #隐层层数
TIMESTEPS = 12  #一个时间步中折叠的递归步数
TRAINING_STEPS = 2000  #训练轮数
BATCH_SIZE = 128  #训练批尺寸

#样本生成函数
def generate_data(seq):
    X = []#初始化输入序列X
    Y= []#初始化输出序列Y
    '''生成连贯的时间序列类型样本集，每一个X内的一行对应指定步长的输入序列，Y内的每一行对应比X滞后一期的目标数值'''
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i:i + TIMESTEPS]])#从输入序列第一期出发，等步长连续不间断采样
        Y.append([seq[i + TIMESTEPS]])#对应每个X序列的滞后一期序列值
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


#构建lstm模型主体
'''定义LSTM cell组件，该组件将在训练过程中被不断更新参数'''
def LstmCell():
    lstm_cell = rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)#
    return lstm_cell

'''定义LSTM模型'''
def lstm_model(X, y):
    '''以前面定义的LSTM cell为基础定义多层堆叠的LSTM，这里只有1层'''
    cell = rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])
    '''将已经堆叠起的LSTM单元转化成动态的可在训练过程中更新的LSTM单元'''
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    '''根据预定义的每层神经元个数来生成隐层每个单元'''
    output = tf.reshape(output, [-1, HIDDEN_SIZE])
    '''通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构'''
    predictions = tf.contrib.layers.fully_connected(output, 1, None)
    '''统一预测值与真实值的形状'''
    labels = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])
    '''定义损失函数，这里为正常的均方误差'''
    loss = tf.losses.mean_squared_error(predictions, labels)
    '''定义优化器各参数'''
    train_op = tf.contrib.layers.optimize_loss(loss,tf.contrib.framework.get_global_step(),
                                      optimizer='Adagrad',learning_rate=0.6)
    '''返回预测值、损失函数及优化器'''
    return predictions, loss, train_op

'''载入tf中仿sklearn训练方式的模块'''
learn = tf.contrib.learn 


#模型保存
'''初始化LSTM模型，并保存到工作目录下以方便进行增量学习'''
regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir='Models/model_1'))

#数据处理
'''对原数据进行尺度缩放'''
data = data_processing(data)
'''将4100个数据来作为训练样本'''
train_X, train_y = generate_data(data[0:4100])
'''将剩余数据作为测试样本'''
test_X, test_y = generate_data(data[4089:-1])

#训练数据
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

#预测测试样本
'''利用已训练好的LSTM模型，来生成对应测试集的所有预测值'''
predicted = np.array([pred for pred in regressor.predict(test_X)])
'''绘制反标准化之前的真实值与预测值对比图'''
p = plt.figure(figsize=(12,8))
ax = p.add_subplot(1,2,1)
plt.plot(predicted, label='预测值')
plt.plot(test_y, label='真实值')
plt.title('反标准化之前')
plt.legend()
plt.show()

'''自定义反标准化函数'''
def scale_inv(raw_data,scale=True):
    '''读入原始数据并转为list'''
    data = pd.read_excel('澳元-人民币.xls',header = 0, sheetname='Sheet1')
    data = data.iloc[:, 4].tolist()
    if scale == True:
        return raw_data*np.std(data)+np.mean(data)
    else:
        return raw_data*(np.max(data)-np.min(data))+np.min(data)
    
sp = scale_inv(predicted)
sy = scale_inv(test_y)

'''绘制反标准化之后的真实值与预测值对比图'''
ax = p.add_subplot(1,2,2)
plt.plot(sp, label='预测值')
plt.plot(sy, label='真实值')
plt.title('反标准化之后')
plt.legend()
plt.show()


'''将预测值与前面的值连起来画图，并与原始图做对比'''
p = plt.figure(figsize=(16,9))
ax = p.add_subplot(1,2,1)
plt.plot(time[4102:-1],sp)
plt.plot(time[0:4100],scale_inv(data[0:4100]))
plt.title('预测图')

ax = p.add_subplot(1,2,2)
plt.plot(time,scale_inv(data))
plt.title('原图')
plt.show()

#计算准确率
acc_num = 0
for i in range(len(sy)):
    if (abs(sp[i]-sy[i])) < 0.05:  #误差小于0.05，则算准确
        acc_num += 1
print('准确率为：',acc_num/len(sp))


#预测未来一个月的值
day=30
l=len(data)
for i in range(day):
    P=[]
    P.append([data[l-TIMESTEPS-1+i:l-1+i]])
    P=np.array(P, dtype=np.float32)
    pre=regressor.predict(P)
    data=np.append(data,pre)
pre=data[len(data)-day:len(data)+1]
print(pre)

#反标准化的值
print(scale_inv(pre))

#预测图
plt.figure()
plt.plot(scale_inv(pre))
plt.title('预测未来一个月的值')
plt.show()