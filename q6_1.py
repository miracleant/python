# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model


###将Feature，Sample和Martix提取出来
def fLoadDataMatrix(filename):#fn-filename
    matrix=[]
    sample=[]
    file=open(filename)
    #第一行数据存入feature
    feature = file.readline().strip('\n').split('\t')
    for line in file:
        sample.append(line.split('\t')[0])
        matrix.append(line.split('\t')[1:])
    xmatrix = np.array(matrix, dtype=float)
    yfeature=np.array(feature)
    return yfeature,sample,xmatrix


###转置
def Transpose(Sample,Feature,Matrix,n):
    x=[]
    for i in range(0,n):
        name_index=Sample.index(Sample[i])#返回行号（第多少行）
        x.append(Matrix[name_index])

    x=np.array(x)
    y=Feature[1:]
    x=x.transpose()#转置x，将原来的行变为列，因为测试的数据是按列读取的！
    return x,y


###公式
def calc(predict,real):
    TP=0.0
    TN=0.0
    FP=0.0
    FN=0.0
    for each in range(len(real)):
        if (predict[each]==1)and(real[each]==1):
            TP+=1
        elif (predict[each]==1)and(real[each]==0):
            FP+=1
        elif (predict[each]==0)and(real[each]==0):
            TN+=1
        else:
            FN+=1
    print('TP:'+str(TP)+' FP:'+str(FP)+' TN:'+str(TN)+' FN:'+str(FN))
    if (TP+FN!=0):
        sn=TP/(TP+FN)
    else:
        sn=0
    if (TN+FP!=0):
        sp=TN/(TN+FP)
    else:
        sp=0
    if (TP+FP+TN+FN!=0):
        acc=(TP+TN)/(TP+FP+TN+FN)
    else:
        acc=0
    avc=(sn+sp)/2
    if ((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN)!=0):
        mcc=(TP*TN-FP*FN)/(math.sqrt((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN)))
    else:
        mcc=0
    return sn,sp,acc,avc,mcc


###SVM
def SVM(x_train,x_test,y_train,y_test):
    clf = svm.SVR()
    clf.fit(x_train,y_train)
    predicted=clf.predict(x_test)
    print(np.mean(predicted==y_test))
    return calc(predicted,y_test)


###NBayes
def NB(x_train,x_test,y_train,y_test):
    #调用MultinomialNB分类器
    clf = MultinomialNB().fit(x_train, y_train)
    predicted = clf.predict(x_test)
    print(np.mean(predicted==y_test))
    return calc(predicted,y_test)

###Dtree
def DT(x_train,x_test,y_train,y_test):
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train, y_train)
    predicted= clf.predict(x_test)
    print (np.mean(predicted==y_test))
    return calc(predicted,y_test)

###KNN
def KNN(x_train,x_test,y_train,y_test):
    clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    print(np.mean(predicted==y_test))
    return calc(predicted,y_test)

def LASSO(x_train,x_test,y_train,y_test):
    clf=linear_model.Lasso(alpha = 0.1)
    clf.fit(x_train,y_train)
    predicted=clf.predict(x_test)
    print(np.mean(predicted ==y_test))
    return calc(predicted,y_test)

Feature = []
Sample = []
Matrix = []
(Feature, Sample, Matrix) = fLoadDataMatrix('C:\Users\huayra\Desktop\python\ALL2.txt')


label=[]
rank_test=[1,10,50,100]
plt.figure(figsize=[11,11])
for i in range(4):
    x,label=Transpose(Sample,Feature,Matrix,rank_test[3])#此处返回了预测的X和y
    label=np.array(label)
    y=np.zeros(label.shape)
    y[label=='FALSE']=1#当feature为POS时，将其y值赋值为1
    # 加载数据集，切分数据集80%训练，20%测试
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



    ##绘制直方图
    X = [0, 2, 4, 6, 8]
    x_labels = ['Sn', 'Sp', 'ACC', 'AVC', 'MCC']
    knn = KNN(x_train, x_test, y_train, y_test)
    nb = NB(x_train, x_test, y_train, y_test)
    svm1 = SVM(x_train, x_test, y_train, y_test)
    dt = DT(x_train, x_test, y_train, y_test)
    lasso = LASSO(x_train, x_test, y_train, y_test)
    plt.subplot(2, 2, i + 1)#4个子图，绘制在第i+1个里面


    plt.bar([x + 0.2 for x in X], knn, 0.2, color='red', label='KNN')
    plt.bar([x + 0.4 for x in X], nb, 0.2, color='blue', label='NB')
    plt.bar([x + 0.6 for x in X], svm1, 0.2, color='gray', label='SVM')
    plt.bar([x + 0.8 for x in X], dt, 0.2, color='pink', label='DTree')
    plt.bar([x + 1.0 for x in X], lasso, 0.2, color='yellow', label='Lasso')
    plt.title('RANK ' + str(rank_test[i]))
    plt.xlabel('Sn             Sp             ACC             AVC             MCC')
    plt.legend(loc='upper right')
plt.show()



