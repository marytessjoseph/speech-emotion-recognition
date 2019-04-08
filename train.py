from python_speech_features import mfcc,logfbank
# from python_speech_features import logfbank
import scipy.io.wavfile as wav
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

mfcc_list = []
fbank_list = []
label_list = []
# i=0

## reading each file ina loop and extracting features, taking average across rows and appending to the list
for filename in glob.glob('/Users/jobilj/Documents/Pro/**/*.wav',recursive=True):
    # print(filename)
    (rate, sig) = wav.read(filename)
    mfcc_feat = mfcc(sig, rate)
    fbank_feat = logfbank(sig, rate)
    mfcc_list.append(np.mean(mfcc_feat, axis=0))
    fbank_list.append(np.mean(fbank_feat, axis=0))
    label_list.append(filename.split('/')[6])

## Merging all features and labels into a data frame
mfcc_names,fbank_names = [],[]
for i in range(len(mfcc_list[1])):
    mfcc_names.append('mfcc_'+str(i))
for i in range(len(fbank_list[1])):
    fbank_names.append('fbank_'+str(i))

data = pd.concat([pd.DataFrame(mfcc_list,columns=mfcc_names),pd.DataFrame(fbank_list,columns=fbank_names),
                  pd.DataFrame(label_list,columns=['label'])],axis=1)

## to see the number of records for each emotion
data.label.value_counts()
##to see top five rows
data.head()
##to see summary of data
data.describe()

## splitting data
train,test = train_test_split(data,test_size=0.15,stratify=data.label,random_state=12345)

## creating X and Y for training and testing
Y_train = train.pop('label')
X_train = train

Y_test = test.pop('label')
X_test = test

###Training the model

C = 1.0 # SVM regularization parameter
model = svm.SVC(kernel='rbf', C=1,gamma='auto').fit(X_train, Y_train)
#model = RandomForestClassifier(n_estimators=50).fit(X_train, Y_train)


###Evaluating the model
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(Y_test, y_pred)
print(conf_mat)
print(classification_report(Y_test, y_pred))
print(accuracy_score(Y_test,y_pred))

