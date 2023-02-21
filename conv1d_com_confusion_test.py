# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.utils import np_utils,plot_model
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from scikeras.wrappers import KerasClassifier
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import model_from_json

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# feature_size = 246
feature_size = 5

df = pd.read_csv("data_com.csv")   #读取时默认把第一行当做属性
print("df_size : ",len(df))
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
    model.add(Conv1D(16, 1, activation='relu'))
    model.add(MaxPooling1D(1))
    model.add(Conv1D(16, 1, activation='relu'))
    model.add(Conv1D(16, 1, activation='relu'))
    model.add(MaxPooling1D(1))
    model.add(Conv1D(16, 1, activation='relu'))
    model.add(Conv1D(16, 1, activation='relu'))
    model.add(MaxPooling1D(1))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    plot_model(model, to_file='./model_classifier.png', show_shapes=True)
    print(model.summary())
    adam = optimizers.Adam(lr=0.0004)
    model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=400, batch_size=16, verbose=1)
estimator.fit(X_train, Y_train)
 
# 混淆矩阵定义
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,('level1','level2','level3','level4','level5'))
    plt.yticks(tick_marks,('level1','level2','level3','level4','level5'))
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('true')
    plt.xlabel('predict')
    plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
    plt.show()
 
# seed = 42
# np.random.seed(seed)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# result = cross_val_score(estimator, X, Y_onehot, cv=kfold)
# print("Accuracy of cross validation, mean %.2f, std %.2f\n" % (result.mean(), result.std()))
 
# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predicted = model.predict(x_val)
    predictions = np.argmax(predicted,axis=1)
    truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))
 
# 将其模型转换为json
model_json = estimator.model.to_json()
with open("./model.json",'w')as json_file:
    json_file.write(model_json)# 权重不在json中,只保存网络结构
estimator.model.save_weights('model.h5')
 
# 加载模型用做预测
json_file = open("./model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 分类准确率
print("The accuracy of the classification model:")
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))
# 输出预测类别
predicted = loaded_model.predict(X)
predicted_label = np.argmax(predicted,axis=1)
print("predicted label:\n " + str(predicted_label))
#显示混淆矩阵
plot_confuse(estimator.model, X_test, Y_test)
 
# 可视化卷积层
# visual(estimator.model, X_train, 1)