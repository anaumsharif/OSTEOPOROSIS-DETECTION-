# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import svm
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from numpy import ndarray
from sklearn import utils
from sklearn import preprocessing
# from dask.distributed import client
from StandardScaler.transform import scaler
from distributed import *
from dask.distributed import *
import skelm
# from large_elm import LargeELMRegressor
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score , classification_report,confusion_matrix ,ConfusionMatrixDisplay
from sklearn.svm import SVR
# reading datasets
df = pd.read_excel(r"C:\Users\HP\CODE\PYTHON\PYCHARM\OSTEOPOROSIS\Dataset.xlsx")
print(df)

print("\n INFO : \n",df.info())
print("\n SHAPE :",df.shape)
print("\n HEAD :\n",df.head)
print("\n ISNULL & SUM :\n",df.isnull().sum())
print("\n COLUMNS : ",df.columns)
print("\n DATA TYPE: \n ",df.dtypes)
print("\n DATA TYPE: \n ",type(df))
print("\n Count of columns in the data is: ", len(df.columns))
print("\n Count of rows in the data is: ", len(df))
print("\n DESCRIBE :\n ",df.describe())
print("\n FREQUENCY COUNT : \n",df['FREQUENCY'].value_counts())
print("\n GROUP-BY: \n",df.groupby('FREQUENCY').mean())
print("\n FREQUENCY: \n",df['FREQUENCY'])

# loop to print all rows of a specific column
data = df.loc[:,'FREQUENCY']
def convert(data):
    li = np.array(data.split(','))
    return li
data1 =[]
# converting strings into float
# ['115,115,113'] to  ['115','115','113']
for key,value in data.items():
    # print(value)
    df_new = convert(value)
    df_new = df_new.tolist()
    data1.append(df_new)
print(data1)
print(type(data1))
print(len(data1))
# ['115','115','113'] to [115.0,115.0,113.0]
data2 =[]
for i in range (len(data1)):
    for j in range (len(data1[i])):
        a = data1[i][j]
        a = float(a)
        data2.append(a)

print(data2)
print(len(data2))
#alternate way using np
# data3 = np.asarray(data2)
# print(data3)
# # data3 = data3.astype

# data3 = data2.np.reshape(1000,3)
# print(data_new)
def nest_list (data2,rows ,columns):
    result =[]
    start =0
    end = columns
    for i in range(rows):
        result.append(data2[start:end])
        start+= columns
        end += columns
    return result
# data_new = []
#
# data3 = np.asarray(data2)
# print(data3)
# # data3 = data3.astype

# data3 = data2.np.reshape(1000,3)
# print(data_new)
for i in data2:
    data_new = nest_list(data2,1000,3)
    data_new.append(data_new)
print(data_new)
print(len(data_new))
print(data_new[1000])
print(data_new[999])
data_new.pop(1000)
print(len(data_new))
print(type(data_new))
print(type(data_new[0][0]))
print(data_new[0])

# removing the string column and inserting the float column in dataframe
df.drop('FREQUENCY',axis=1,inplace = True)
df['FREQUENCY']= data_new

# performing standard deviation
std_dev = []
for i in range (len(data_new)):
    a = np.std(data_new[i])
    std_dev.append(a)
print(std_dev)
print(type(std_dev))
print(len(std_dev))
std_dev1 = list(map(int,std_dev))
print(std_dev1)
print(type(std_dev1))
print(len(std_dev1))
#  rounding off converting a std_dev list to string
# std_dev1 = [ '%.2f' % elem for elem in std_dev ]
# print(std_dev1)
# print(len(std_dev1))
# # converting a std_dev list to string
# std_dev2 = list(map(str,std_dev1))
# print(std_dev2)
# print(type(std_dev2))
# print("length of std_dev2",len(std_dev2))
# # Default Data type of Array
# Data_type = object
#
# std_dev3= numpy.array(std_dev2, dtype= Data_type)
# inserting a new column in dataframe
df['STD_DEV'] = std_dev1
print(df)
#
# # converting a avg dataframe column to  list
# avg = df['avg'].tolist()
# print(type(avg))
# print(len(avg))
#
# # converting avg list to string
# avg1 = list(map(str,avg))
# print(avg1)
# print(type(avg1))
# print("length of avg1",len(avg1))



# # Default Data type of Array
# Data_type = object
# avg2= numpy.array(avg1, dtype=Data_type)
#
# # removing the float column and inserting the string column in dataframe
# df.drop('avg',axis=1,inplace = True)
# df['avg']= avg2
# print(df)


# checking counts of columns in dataframe
print("\n\n H/O INJURY/SURGERY COUNT : \n",df['H/O INJURY/SURGERY'].value_counts())
print("\n\n ASSO MEDICAL PROB COUNT : \n",df['ASSO MEDICAL PROB'].value_counts())
print("\n\n DRUG HISTORY COUNT : \n",df['DRUG HISTORY'].value_counts())
print("\n\n SEX: \n",df['SEX'].value_counts())
print("\n\n STD_DEV: \n",df['STD_DEV'].value_counts())

# removing or deleting the unnecessary data from the dataframe
df.drop(['DATE','NAME','SI.NO','FREQUENCY'],axis=1,inplace =True)
print(df)
# mapping the elements of column in dataframe
df['SEX'] = df['SEX'].map({'Male':0,'male':0, 'Female':1,'female':1})
df['ASSO MEDICAL PROB'] =df['ASSO MEDICAL PROB'].map({'no' :0,'No':0 ,'yes(diabetes)':1,'yes (diabetes)':1,'yes(bp)':2,'yes(diabetes,bp)':3,'Yes(Diabetes,bp)':3,'yes(bp dabetes)':3,'kidney stone':4,'yes(increase in heart rate)':5,'Yes(Diabetes,Blockage in Heart)':6,'yes (diabetes,heart blockage)':6,'yes(diabetes,kidney stone)':7})
df['H/O INJURY/SURGERY']= df['H/O INJURY/SURGERY'].map({'no':0,'vericose vein surgery':1,'uteres removal':2,'kidney stone opreration':3 ,'uterus surgery':4,'yes(diverticulities)':5,'shouler surgery':6,'knee surgery':7,'yes(open heart surgery)':8})
df['DRUG HISTORY'] =df['DRUG HISTORY'].map({'no':0,'yes':1,'yes(ecosprin)':2})
print(df)
df.to_csv("Clean_Dataset.csv")
print(df.dtypes)
print(df.head())
print("MISING VALUES :\n",df.isnull().sum())
print(df.describe(include = 'all'))
# seperating independent and dependent features
x = df.drop('STD_DEV', axis=1)
y = df['STD_DEV']
# y = y.astype('int')

print(x.head())
print(y.head())


# splitting the data into training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# check the shape of the training and the testing data
print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))
print("Size of x_test:", (x_test.shape))
print("Size of y_test:", (y_test.shape))


ss = StandardScaler()
print(ss.fit(x_train))

x_train= ss.fit_transform(x_train)
x_test = ss.transform(x_test)
print("\n X_TRAIN :\n",x_train)
print("\n X_TEST :\n",x_test)
model = LinearSVC()
# model1= model.fit(x_train,y_train)

model.fit(x_train,y_train)
x_train_pred = model.predict(x_train)
train_data_acc = accuracy_score(y_train,x_train_pred)
print("accuracy of training data :",train_data_acc*100.)
x_test_pred = model.predict(x_test)
test_data_acc = accuracy_score(y_test,x_test_pred)
print("accuracy of testing data :", test_data_acc*100)


#loading library
# import joblib
# import sklearn.externals
# joblib.dump(model,'model_j')
# m_jlib = joblib.load('model_jlib')




import pickle
with open('model.pkl','wb') as files :
    pickle.dump(model,files)
with open('model.pkl','rb') as f:
    lr = pickle.load(f)
#

# def ValuePredictor(to_predict_osteoporosis):
#     to_predict = np.array(to_predict_osteoporosis).reshape(-1)
#     loaded_model = pickle.load(open("model.pkl", "rb"))
#     result = loaded_model.predict(to_predict)
#     return result[0]
#
#
# @app.route('/result', methods=['POST'])
# def result():
#     if request.method == 'POST':
#         to_predict_list = request.form.to_dict()
#         to_predict_list = list(to_predict_list.values())
#         to_predict_list = list(map(int, to_predict_list))
#         result = ValuePredictor(to_predict_list)
#         if int(result) == 1:
#             prediction = 'You have significant risk of having  OSTEOPOROSIS  '
#         else:
#             prediction = ' you have a HEALTHY bone density '
#         return render_template("result.html", prediction=prediction)