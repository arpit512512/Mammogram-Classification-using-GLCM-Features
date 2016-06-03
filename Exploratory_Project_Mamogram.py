
# coding: utf-8

# In[1]:

from skimage import io
from skimage.viewer import ImageViewer
import pandas as pd
from sklearn import linear_model
import math
import numpy as np
import skimage
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import copy


# In[4]:

path1 = "/media/arpit/New Volume/ML/project/TA/Final_Dataset/"
path2 = "/media/arpit/New Volume/ML/project/final_results/"
img_data = pd.read_csv(path1+'Final_data.csv',header=None)
img_data = img_data[:]
data_size = img_data.shape[0]
img_data.head()


# In[5]:

def get_glcm_features(angle_code, dist,Z): ## angle_code 0-->1, 45-->2, 90-->3, 135-->4, 180-->5
    X=copy.deepcopy(Z)
    fa1,fb1,fc1,fd1 = np.zeros(data_size),np.zeros(data_size),np.zeros(data_size),np.zeros(data_size)
    img_name = list(img_data[0])
    for i in range(len(img_data)) :
        name = str(img_name[i])+".pgm"
        I = io.imread(path1+name)
        if angle_code==1:
            glcm = skimage.feature.greycomatrix(I, [dist], [0], normed=True)
        elif angle_code==2:
            glcm = skimage.feature.greycomatrix(I, [dist], [-dist], normed=True)
        elif angle_code==3:
            glcm = skimage.feature.greycomatrix(I, [0], [-dist], normed=True)
        elif angle_code==4:
            glcm = skimage.feature.greycomatrix(I, [-dist], [-dist], normed=True)
        elif angle_code==5:
            glcm = skimage.feature.greycomatrix(I, [-dist], [0], normed=True)

        fa1[i] = skimage.feature.greycoprops(glcm, 'contrast')[0][0]
        fb1[i] = skimage.feature.greycoprops(glcm, 'energy')[0][0]
        fc1[i] = skimage.feature.greycoprops(glcm, 'homogeneity')[0][0]
        fd1[i] = skimage.feature.greycoprops(glcm, 'correlation')[0][0]
    X['contrast_ang'+str(angle_code)+'_d'+str(dist)],X['energy_ang'+str(angle_code)+'_d'+str(dist)],X['homogeneity_ang'+str(angle_code)+'_d'+str(dist)],X['correlation_ang'+str(angle_code)+'_d'+str(dist)]=fa1,fb1,fc1,fd1
    return X


# In[6]:

def indices_list(l) :
    scores = zip(range(len(l)),l)
    scores = sorted(scores, key=lambda a:a[1])
    scores = scores[::-1]
    scores = map(lambda x:x[0],scores)
    indices = scores[:4]
    return indices


# In[22]:

def get_accuracy_cvset2(model,x_train,x_test,y_train,y_test) :
    model.fit(x_train,y_train)
    ypred = model.predict(x_test)
    y_test=np.array(y_test)
    mis=0
    for i in range(len(ypred)) : mis += abs(ypred[i]-y_test[i])
    #print len(ypred[ypred==1])
    #return (len(ypred)-mis)/(len(ypred)+0.00)
    conf_mtr = confusion_matrix(y_test, ypred)
    accu = float(conf_mtr[0][0]+conf_mtr[1][1])/(len(ypred))
    sens = float(conf_mtr[1][1])/float(conf_mtr[1][0]+conf_mtr[1][1])
    spec = float(conf_mtr[0][0])/float(conf_mtr[0][0]+conf_mtr[0][1])
    return (accu,sens,spec,conf_mtr,ypred)


# In[8]:

def get_accuracy_cvset(model,x_train,x_test,y_train,y_test) :
    model.fit(x_train,y_train)
    ypred = model.predict(x_test)
    y_test=np.array(y_test)
    mis=0
    for i in range(len(ypred)) : mis += abs(ypred[i]-y_test[i])
    #print len(ypred[ypred==1])
    return (len(ypred)-mis)/(len(ypred)+0.00)


# In[9]:

def get_accuracy(X,y,trees) :
    model = RandomForestClassifier(n_estimators=trees)
    return cross_validation.cross_val_score(model,X,y,cv=5).mean()


# In[10]:

Y = img_data[3].apply(lambda x : 0 if x=='N' else 1)
X = pd.DataFrame()
X2 = pd.DataFrame()
L=[]
for i in range(1,4) :
    l=[]
    for j in range(1,30) : 
        X = get_glcm_features(i,j,pd.DataFrame())
        X2 = get_glcm_features(i,j,X2)
        l.append(get_accuracy(X,Y,100))
    L.append(l)


# In[63]:

# plt.xlim(0,31)
# plt.ylim(40,100)
# plt.xlabel("Pixel Distance in(px)")
# plt.ylabel("Accuracy in(%)")
# plt.xticks(np.arange(0,31,5))

# for i in range(len(L)):
#     if(i==0):
#         plt.plot(range(1,30),L[i],label='$%i^\circ$'%0,linestyle='--')
#     if(i==1):
#         plt.plot(range(1,30),L[i],label='$%i^\circ$'%45,linewidth='2')
#     if(i==2):
#         plt.plot(range(1,30),L[i],label='$%i^\circ$'%90,color='y')
#     plt.scatter(range(1,30),L[i],facecolors='r')
#     #plt.annotate(str(i+1), xy=(i+10, L[i][i+10]), xytext=(i+10, L[i][i+10]))

# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1),
#       ncol=3, fancybox=True, shadow=True)
# plt.show()


# In[11]:

X = pd.DataFrame()
for i in range(1,4):
    indices = indices_list(L[i-1])
    for j in indices:
        X = get_glcm_features(i,j+1,X)
print get_accuracy(X,Y,100)
X.head()


# In[12]:

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,Y, test_size=0.25,random_state=3)


# In[13]:

from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
model6 = AdaBoostClassifier(n_estimators=500)
model6.fit(X_train,y_train)
x = 100*model6.feature_importances_+0.1
print x


# In[14]:

plt.bar(range(X.shape[1]), x)
plt.xticks(np.arange(0,51,5))
plt.xlabel("Features")
# plt.ylim()
plt.xlim(0,50)
plt.ylabel("Feature score")
plt.show()


# In[15]:

feature_threshold=6
Q=X_train.iloc[:,x>=feature_threshold]
print Q.shape
Q.head()#graph analysis


# In[53]:

from sklearn.decomposition import PCA
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,Y, test_size=0.5,random_state=3)
#X_test=X
#y_test=Y
Qtrain=X_train.iloc[:,x>=feature_threshold]
Qtest=X_test.iloc[:,x>=feature_threshold]
pca = PCA(n_components=15).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
model = GradientBoostingClassifier(n_estimators=100)
model2 = RandomForestClassifier(n_estimators=100)
print get_accuracy_cvset(model,X_train_pca,X_test_pca,y_train,y_test)
print get_accuracy_cvset(model,X_train,X_test,y_train,y_test)
print get_accuracy_cvset(model,Qtrain,Qtest,y_train,y_test)
print("rf")
print get_accuracy_cvset(model2,X_train_pca,X_test_pca,y_train,y_test)
print get_accuracy_cvset(model2,X_train,X_test,y_train,y_test)
print get_accuracy_cvset(model2,Qtrain,Qtest,y_train,y_test)


# In[58]:

model  = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
ypred = model.predict(X_test)
y_test=np.array(y_test)
mis=0
for i in range(len(ypred)) : mis += abs(ypred[i]-y_test[i])
    #print len(ypred[ypred==1])
print (len(ypred)-mis)/(len(ypred)+0.00)


# In[17]:

scores = zip(range(len(x)),x)
scores = sorted(scores, key=lambda a:a[1])
scores = scores[::-1]
scores = map(lambda x:x[0],scores)


# In[23]:

accu_no=[]
spec_no=[]
sens_no=[]
conf_no=[]
ypred_no=[]
mx=0
for i in range(1,len(scores)+1):
    Qtrain=X_train.iloc[:,scores[:i]]
    Qtest=X_test.iloc[:,scores[:i]]
    model2 = RandomForestClassifier(n_estimators=100)
    accu,sens,spec,conf,ypred=get_accuracy_cvset2(model2,Qtrain,Qtest,y_train,y_test)
    conf_no.append(conf)
    accu_no.append(accu)
    sens_no.append(sens)
    spec_no.append(spec)
    ypred_no.append(ypred)
    mx=max(mx,accu)
print mx


# In[24]:

for i in range(len(accu_no)):
    accu_no[i]=100*accu_no[i]
    spec_no[i]=100*spec_no[i]
    sens_no[i]=100*sens_no[i]


# In[40]:

plt.xlabel("Number of features")
plt.xlim(0,50)
plt.ylim(50,101)
plt.ylabel("Accuracy in (%)")
plt.yticks(np.arange(50, 101,5))
plt.xticks(np.arange(0, 51,5))
plt.scatter(range(1,len(scores)+1),accu_no[:],facecolors='r')
plt.plot(range(1,len(scores)+1),accu_no)
# plt.scatter(range(1,len(scores)+1),spec_no,facecolors='r')
# plt.plot(range(1,len(scores)+1),spec_no,'y')
# plt.scatter(range(1,len(scores)+1),sens_no,facecolors='r')
# plt.plot(range(1,len(scores)+1),sens_no,'g')
plt.show()


# In[62]:

#zip(range(1,1+len(accu_no)),accu_no,sens_no,spec_no)



