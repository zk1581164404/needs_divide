# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
# from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils,plot_model
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from scikeras.wrappers import KerasClassifier
import math
from matplotlib.pyplot import MultipleLocator
# feature_size = 246
feature_size = 5


# 载入数据
# df = pd.read_csv("数据集-用做分类.csv")
df = pd.read_csv("data_test.csv")
X = np.expand_dims(df.values[:, :-1].astype(float), axis=2)
Y = df.values[:, feature_size]
 
# 湿度分类编码为数字
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot = np_utils.to_categorical(Y_encoded)
 
# 划分训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.3, random_state=0)

test_bound = len(X_train)
valid_bound = int(test_bound * 0.8)
data_size = len(Y)

print(test_bound,valid_bound,data_size)

# 定义神经网络
def baseline_model():
    model = Sequential()
    model.add(Conv1D(16, 1, input_shape=(feature_size, 1)))
    model.add(Conv1D(16, 1, activation='tanh'))
    model.add(MaxPooling1D(1))
    model.add(Conv1D(16, 1, activation='tanh'))
    model.add(Conv1D(16, 1, activation='tanh'))
    model.add(MaxPooling1D(1))
    model.add(Conv1D(16, 1, activation='tanh'))
    model.add(Conv1D(16, 1, activation='tanh'))
    model.add(MaxPooling1D(1))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    plot_model(model, to_file='./model_classifier.png', show_shapes=True)
    print(model.summary())
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model
    # model = Sequential()
    # model.add(Conv1D(16, 1, input_shape=(feature_size, 1)))
    # model.add(Conv1D(16, 1, activation='tanh'))
    # model.add(MaxPooling1D(1))
    # model.add(Conv1D(16, 1, activation='tanh'))
    # model.add(Conv1D(16, 1, activation='tanh'))
    # model.add(MaxPooling1D(1))
    # model.add(Conv1D(16, 1, activation='tanh'))
    # model.add(Conv1D(16, 1, activation='tanh'))
    # model.add(MaxPooling1D(1))
    # model.add(Flatten())
    # model.add(Dense(5, activation='softmax'))
    # plot_model(model, to_file='./model_classifier.png', show_shapes=True)
    # print(model.summary())
    # model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    # return model

model = baseline_model()

print(model.summary())   

# 训练分类器

from timeit import default_timer as timer
start = timer()

history = model.fit(X_train,
                    Y_train,
                    batch_size=128,
                    epochs=100,
                    validation_split=0.2,
                    verbose=2)
                    
end = timer()
print(end - start)

history_dict = history.history
history_dict.keys()

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
loss_values50 = loss_values[0:150]
val_loss_values50 = val_loss_values[0:150]
epochs = range(1, len(loss_values50) + 1)
plt.plot(epochs, loss_values50, 'b',color = 'blue', label='Training loss')
plt.plot(epochs, val_loss_values50, 'b',color='red', label='Validation loss')
plt.rc('font', size = 18)
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.xticks(epochs)
fig = plt.gcf()
fig.set_size_inches(15,7)


x_major_locator=MultipleLocator(10)
#把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator=MultipleLocator()
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
# ax.yaxis.set_major_locator(y_major_locator)


#fig.savefig('img/tcstest&validationlosscnn.png', dpi=300)
plt.show()

# In[11]:

# 画出训练集和验证集的误差图像

mae = history_dict['accuracy']  #这里改成mae
vmae = history_dict['val_accuracy']  #对应这里就要改成val_mae
epochs = range(1, len(mae) + 1)
plt.plot(epochs, mae, 'b',color = 'blue', label='Training error')
plt.plot(epochs, vmae, 'b',color='red', label='Validation error')
plt.title('Training and validation error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.xticks(epochs)
fig = plt.gcf()
fig.set_size_inches(15,7)  #设置图像大小
x_major_locator=MultipleLocator(10)
#把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator=MultipleLocator()
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#fig.savefig('img/tcstest&validationerrorcnn.png', dpi=300)
plt.show()

model.metrics_names
trainScore = model.evaluate(X_train, Y_train, verbose=0)
testScore = model.evaluate(X_test, Y_test, verbose=0)

p = model.predict(X_test)

# 画出真实值和测试集的预测值之间的误差图像
plt.plot(np.argmax(p, axis=1),color='red', label='prediction')
plt.plot(np.argmax(Y_test, axis=1),color='blue', label='y_test')
# print("Y_test : ",np.argmax(Y_test, axis=1))
# print("p : ",np.argmax(p, axis=1))
plt.xlabel('No. of Trading Days')
plt.ylabel('Close Value (scaled)')
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(15, 5)
x_major_locator=MultipleLocator(20)
#把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator=MultipleLocator()
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#fig.savefig('img/tcstestcnn.png', dpi=300)
plt.show()
p1= model.predict(X_train)
print("p1.size == ",len(p1))
#验证集相关  这里都改成超参数吧
plt.plot(np.argmax(p1[:valid_bound],1),color='red', label='prediction on training samples')
# x = np.array(range(848,1060))
x = np.array(range(valid_bound,test_bound))   #数据集的数据维度  需要自己更改一下

plt.plot(x,np.argmax(p1[valid_bound:test_bound],1),color = 'magenta',label ='prediction on validating samples')
plt.plot(np.argmax(Y_train,1),color='blue', label='Y_train')

plt.xlabel('No. of Trading Days')
plt.ylabel('Close Value (scaled)')
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(20,10)
#fig.savefig('img/tcstraincnn.png', dpi=300)
plt.show()


from sklearn.metrics import mean_absolute_error

print('Trainscore RMSE \tTrain Mean abs Error \tTestscore Rmse \t Test Mean abs Error')
print('%.9f \t\t %.9f \t\t %.9f \t\t %.9f' % (math.sqrt(trainScore[0]),trainScore[1],math.sqrt(testScore[0]),testScore[1]))

#  训练集、验证集、测试集 之间的比较

Y = np.concatenate((np.argmax(Y_train,1),np.argmax(Y_test,1)),axis = 0)
P = np.concatenate((np.argmax(p1,1),np.argmax(p,1)),axis = 0)
#plotting the complete Y set with predicted values on x_train and x_test(variable p1 & p respectively given above)
#for 
plt.plot(P[:test_bound],color='red', label='prediction on training samples')
#for validating samples
z = np.array(range(valid_bound,test_bound))
plt.plot(z,P[valid_bound:test_bound],color = 'black',label ='prediction on validating samples')
# #for testing samples
x = np.array(range(test_bound,data_size))
plt.plot(x,P[test_bound:data_size],color = 'green',label ='prediction on testing samples(x_test)')


plt.plot(Y,color='blue', label='Y')
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(20,12)
plt.show()

loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=64)
print('## evaluation loss and metrics ##')
print(loss_and_metrics)















# estimator = KerasClassifier(build_fn=baseline_model, epochs=40, batch_size=128, verbose=1)
# estimator.fit(X_train, Y_train)
 
# 卷积网络可视化
# def visual(model, data, num_layer=1):
#     # data:图像array数据
#     # layer:第n层的输出
#     layer = keras.backend.function([model.layers[0].input], [model.layers[num_layer].output])
#     f1 = layer([data])[0]
#     print(f1.shape)
#     num = f1.shape[-1]
#     print(num)
#     plt.figure(figsize=(8, 8))
#     for i in range(num):
#         plt.subplot(np.ceil(np.sqrt(num)), np.ceil(np.sqrt(num)), i+1)
#         plt.imshow(f1[:, :, i] * 255, cmap='gray')
#         plt.axis('off')
#     plt.show()
 
# 混淆矩阵定义
# def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.jet):
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
#     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks,('0%','3%','5%','8%','10%','12%','15%','18%','20%','25%'))
#     plt.yticks(tick_marks,('0%','3%','5%','8%','10%','12%','15%','18%','20%','25%'))
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('真实类别')
#     plt.xlabel('预测类别')
#     plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
#     plt.show()


# seed = 42
# np.random.seed(seed)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# result = cross_val_score(estimator, X, Y_onehot, cv=kfold)
# print("Accuracy of cross validation, mean %.2f, std %.2f\n" % (result.mean(), result.std()))
 
# # 显示混淆矩阵
# def plot_confuse(model, x_val, y_val):
#     predictions = model.predict_classes(x_val)
#     truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
#     conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
#     plt.figure()
#     plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))
 
# # 将其模型转换为json
# model_json = estimator.model.to_json()
# with open(r"C:\Users\316CJW\Desktop\毕设代码\model.json",'w')as json_file:
#     json_file.write(model_json)# 权重不在json中,只保存网络结构
# estimator.model.save_weights('model.h5')
 
# # 加载模型用做预测
# json_file = open(r"C:\Users\Desktop\model.json", "r")
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("model.h5")
# print("loaded model from disk")
# loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # 分类准确率
# print("The accuracy of the classification model:")
# scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
# print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))
# # 输出预测类别
# predicted = loaded_model.predict(X)
# predicted_label = loaded_model.predict_classes(X)
# print("predicted label:\n " + str(predicted_label))
# #显示混淆矩阵
# plot_confuse(estimator.model, X_test, Y_test)
 
# # 可视化卷积层
# # visual(estimator.model, X_train, 1)