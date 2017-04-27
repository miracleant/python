# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB

path = "c:\Users\huayra\Desktop\python\ALL3.txt"
file = open(path)
feature = file.readline().strip('\n').split('\t')
###方便后面的特征向量选择时准确度的返回
result=0

j = 0
df = pd.read_csv(path, sep='\t')
for line in file:
    line = line.strip('\n')
    lines = line.split('\t')
    dataN = []
    dataP = []
    for i in range(1, len(lines)):
        if (feature[i] == 'NEG'):
            dataN.append(float(lines[i]))
        if (feature[i] == 'POS'):
            dataP.append(float(lines[i]))

    tvalue, pvalue = stats.ttest_ind(dataP, dataN)
    df.ix[j, 'pvalue'] = pvalue
    df.ix[j, 'tvalue'] = tvalue
    j = j + 1

####T排序的前10组数据
result10 = pd.DataFrame(df.sort(columns='pvalue').head(10), columns=['mdr', 'pvalue', 'tvalue'])
print (result10)

sortValue = df.sort(columns='pvalue')
result = pd.DataFrame(sortValue.values)

def draw(x, y):
    figure = plt.figure()
    ax1 = figure.add_subplot(111)
    ax1.set_title('result dot picture'+str(x)+'-'+str(y))
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    # 画散点图
    for i in range(len(feature)):
        if (feature[i] == 'NEG'):
            ax1.plot(result.ix[x - 1, i], result.ix[y - 1, i], 'r-o')
        if (feature[i] == 'POS'):
            ax1.plot(result.ix[x - 1, i], result.ix[y - 1, i], 'g-o')
    plt.show()


###公式
def calc(predict,real):
    t_p=0.0
    t_n=0.0
    f_p=0.0
    f_n=0.0
    for each in range(len(real)):
        #实际上是正样本，被判定为正样本
        if (predict[each]==1)and(real[each]==1):
            t_p+=1
        elif (predict[each]==1)and(real[each]==0):
            f_p+=1
        elif (predict[each]==0)and(real[each]==0):
            t_n+=1
        else:
            f_n+=1
    print('t_p:'+str(t_p)+' f_p:'+str(f_p)+' t_n:'+str(t_n)+' f_n:'+str(f_n))
    if (t_p+f_n!=0):
        sn=t_p/(t_p+f_n)
    else:
        sn=0
    if (t_n+f_p!=0):
        sp=t_n/(t_n+f_p)
    else:
        sp=0
    if (t_p+f_p+t_n+f_n!=0):
        acc=(t_p+t_n)/(t_p+f_p+t_n+f_n)
    else:
        acc=0
    avc=(sn+sp)/2
    if ((t_p+f_n)*(t_p+f_p)*(t_n+f_p)*(t_n+f_n)!=0):
        mcc=(t_p*t_n-f_p*f_n)/(math.sqrt((t_p+f_n)*(t_p+f_p)*(t_n+f_p)*(t_n+f_n)))
    else:
        mcc=0
    return sn,sp,acc,avc,mcc

def XValue(num):
    if(num>=0):
        matrix=df.sort_values(by='pvalue').head(num).T.iloc[1:len(feature),0:].as_matrix()
    else:
        matrix=df.sort_values(by='pvalue').tail(num).T.iloc[1:len(feature),0:].as_matrix()
    return matrix


def NB(x_train,x_test,y_train,y_test):
    #调用MultinomialNB分类器
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    result=np.mean(predicted == y_test)
    return calc(predicted, y_test)

###SVM
def SVM(x_train,x_test,y_train,y_test):
    clf = svm.SVC()
    clf.fit(x_train,y_train)
    predicted=clf.predict(x_test)
    print np.mean(predicted == y_test)
    return calc(predicted, y_test)


###Dtree
def DT(x_train,x_test,y_train,y_test):
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train, y_train)
    predicted= clf.predict(x_test)
    print np.mean(predicted == y_test)
    return calc(predicted, y_test)

###KNN
def KNN(x_train,x_test,y_train,y_test):
    clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    print np.mean(predicted == y_test)
    return calc(predicted, y_test)

def LASSO(x_train,x_test,y_train,y_test):
    #clf=linear_model.Lasso(alpha = 0.5)
    clf=linear_model.LassoCV()
    clf.fit(x_train,y_train)
    answer = clf.predict(x_test)
    predicted=answer>0.5
    predicted = np.array(predicted, dtype=int)
    print np.mean(predicted ==y_test)
    return calc(predicted,y_test)

###特征向量选择
def feasel():
    resultNB=[]
    resultKNN=[]
    resultSVM=[]
    resultDT=[]
    resultLASSO=[]
    selection=[2,4,6,8,10,12,14,16,18,20]
    for i in range(len(selection)):
        x=XValue(selection[i])
        label=np.array(feature[1:])
        y=np.zeros(label.shape)
        y[label=='POS']=1
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)
        ###准确度返回
        NB(x_train, x_test, y_train, y_test)
        resultNB.append(result)
        DT(x_train, x_test, y_train, y_test)
        resultDT.append(result)
        SVM(x_train, x_test, y_train, y_test)
        resultSVM.append(result)
        KNN(x_train, x_test, y_train, y_test)
        resultKNN.append(result)
        LASSO(x_train, x_test, y_train, y_test)
        resultLASSO.append(result)


    plt.figure(figsize=(8,4))
    plt.plot(selection,resultNB,"bo-",linewidth=1)
    plt.plot(selection,resultDT,"go-",linewidth=1)
    plt.plot(selection,resultKNN,"ro-",linewidth=1)
    plt.plot(selection,resultSVM,"ko-",linewidth=1)
    plt.plot(selection,resultLASSO,"yo-",linewidth=1)
    plt.legend(['NB','DT','KNN','SVM','LASSO'],loc='upper right')
    plt.xlabel("Incremental feature selection")
    plt.ylabel("Acc")
    plt.show()

###Sn,Sp,ACC,AVC,MCC的计算
def fivePre():
    rank_test = [1, 10, 100, -100]
    plt.figure(figsize=[11, 11])
    for i in range(4):
        x = XValue(rank_test[i])  # 此处返回了预测的X
        label = np.array(feature[1:])
        y = np.zeros(label.shape)
        y[label == 'POS'] = 1  # 当feature为POS时，将其y值赋值为1
        # 加载数据集，切分数据集80%训练，20%测试
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        ###画图
        X = [0, 2, 4, 6, 8]
        knn = KNN(x_train, x_test, y_train, y_test)
        nb = NB(x_train, x_test, y_train, y_test)
        svm1 = SVM(x_train, x_test, y_train, y_test)
        dt = DT(x_train, x_test, y_train, y_test)
        lasso = LASSO(x_train, x_test, y_train, y_test)

        plt.subplot(2, 2, i + 1)
        plt.bar([x + 0.2 for x in X], knn, 0.2, color='red', label='KNN')
        plt.bar([x + 0.4 for x in X], nb, 0.2, color='blue', label='NB')
        plt.bar([x + 0.6 for x in X], svm1, 0.2, color='gray', label='SVM')
        plt.bar([x + 0.8 for x in X], dt, 0.2, color='pink', label='DTree')
        plt.bar([x + 1.0 for x in X], lasso, 0.2, color='yellow', label='Lasso')
        plt.title('test ' + str(rank_test[i]))
        plt.xlabel('Sn             Sp             ACC             AVC             MCC')
        plt.legend(loc='upper right')
    plt.show()


####主调用函数



###画点状图
draw(1,2)
draw(9,10)
draw(1000, 1001)
draw(10000,10001)


feasel()


fivePre()