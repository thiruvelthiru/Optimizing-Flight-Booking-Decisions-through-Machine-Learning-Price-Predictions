import numpy as np
import pandas as pd
import array as ar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,confusion_matrix
import warnings
import pickle
from scipy import stats
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
data = pd.read_excel("/content/drive/MyDrive/Colab Notebooks/Data_Train(1).xlsx")
data.head()


category = ['Airline','Date_of_Journey','Source','Destination','Route','Dep_Time','Arrival_Time','Duration','Total_Stops','Additional_Info','Price']
for i in category:
  print(i,data[i].unique())
  
data.Date_of_Journey=data.Date_of_Journey.str.split('/')
data.Date_of_Journey


data['Data']=data.Date_of_Journey.str[0]
data['month']=data.Date_of_Journey.str[1]
data['year']=data.Date_of_Journey.str[2]
data.Total_Stops.unique()


data.Route=data.Route.str.split('')
data.Route

data['City1']=data.Route.str[0]
data['City2']=data.Route.str[1]
data['City3']=data.Route.str[2]
data['City4']=data.Route.str[3]

data.Dep_Time=data.Dep_Time.str.split(':')
data['Dep_Time_Hour']=data.Dep_Time.str[0]
data['Dep_Time_Mins']=data.Dep_Time.str[1]
data.Arrival_Time=data.Arrival_Time.str.split('')
data['Arrival_Time']=data.Arrival_Time.str[1]
data['Time_of_Arrival']=data.Arrival_Time.str[0]
data['Time_of_arrival']=data.Time_of_Arrival.str.split(':')
data['Arrival_Time_Hour']=data.Time_of_arrival.str[0]
data['Arrival_Time_Mins']=data.Time_of_Arrival.str[1]

data.Duration=data.Duration.str.split(' ')
data['Travel_Hours']=data.Duration.str[0]
data['Travel_Hours']=data['Travel_Hours'].str.split('h')
data['Travel_Hours']=data['Travel_Hours'].str[0]
data.Travel_Hours=data.Travel_Hours
data['Travel_Mins']=data.Duration.str[1]
data.Travel_Mins=data.Travel_Mins.str.split('m')
data.Travel_Mins=data.Travel_Mins.str[0]

data.Total_Stops.replace('non_stop',0,inplace=True)
data.Total_Stops=data.Total_Stops.str.split(' ')
data.Total_Stops=data.Total_Stops.str[0]

data.Additional_Info.unique
array(['No info','In-flight meal not included','No check-in baggage included','1 Short layover','No Info','1 Long layover','Change airports','Business class','Red-eye flight','2 Long layover'])

data.isnull().sum()

data.drop(['City4','City5','City6'],axis=1,inplace=True) 

data.drop(['Data_of_Journey','Route','Dep_Time','Arrival_time','Duration'],axis=1,inplace=True)
data.drop(['Time_of_Arrival'],axis=1,inplace=True)

data.insull().sum()

#Replace Missing Values


data['City3'].fillna('None',inplace=True)
data['Arrival_data'].fillna(data['Data'],inplace=True)
data['Travel_mins'].fillna(0,inplace=True)


data.info()
<class 'pandas.core.frame.DataFrame'>




data.Data=data.Data.astype('int64')
data.Month=data.Month.astype('int64')
data.Year=data.Year.astype('int64')
data.Dep_Time_Hour=data.Dep_Time_Hour.astype('int64')
data.Dep_Time_Hour=data.Dep_Time_Hour.astype('int64')
data.Dep_Time_Mins=data.Dep_Time_Mins.astype('int64')
data.Arrival_date=data.Arrival_date.astype('int64')
data.Arrival_Time_Hour=data.Arrival_Time_Hour.astype('int64')
data.Arrival_Time_Mins=data.Arrival_Time_Mins.astype('int64')



data[data['Travel_Hours']=='5m']

data.drop(index=6474,inplace=True,axis=0)

data.Travel_hous=data.Travel_Hours.astype('int64')


categorical=['Ailine','Souce','Destination','Additional_Info','City1']
numerical=['Total_Stops','Date','Month','Year','Dep_Time_Hou','Dep_Time_Mins','Arrival_date','Arrival_Time_Hour','Arrival_Time_Mins','Travel_Hours','Travel_Mins,]




⁮#LABEL ENCODING


from sklearn.preprocessing import LabelEncode
le=LableEncode()

data.Airline=le.fit_transform(data.Airline)
data.Source=le.fit_transform(data.Source)
data.Destination=le.fit_transform(data.Destination)
data.Total_Stops=le.fit_transform(data.Total_stops)
data.City1=le.fit_transform(data.City1)
data.City2=le.fit_transform(data.City2)
data.City3=le.fit_transform(data.City3)
data.Additional_Info=le.fit_transform(data.Addition_Info)
data.head()



#Output Colums\


data.head()
data=data[['Airline','Source','Destination','Date','Month','Year','Dep_Time_Hour','Dep_Time_Mins','Arrival_date','Arrival_Time_



data.head()


#DESCRIPTIVE STATISTICAL

data.describe()


#VISUAL ANALYSIS

import seaborn as sns
c=1
plt.figure(figsize=(20,45))

for i in categorical:
 plt.subplot(6,3,c)
 sns.countplot(data[i])
 plt.xticks(rotation=90)
 plt.tight_layout(pad=3,0)
 c=c+1
plt.show()


#PRICE COLUM

plt.figure(figsize=(15,8))
sns.distplot(data.Price)

#CHECKING THE CORRELATION USING HEATMAP


sns.heatmap(data.corr(),annot=True)


#OUTLIER DETECTION FOR PRICE

import seaborn as sns
sns.boxplot(data['Price'])

y=data['Price']
x=data.drop(colums=['Price'],axis=1)


from sklearn.preprocessing import StandrdScaler
ss=StandardScaler()

x_scaler=ss.fit_transform(x)


x_scaled=pd.DataFrame(x_scaled,colums=x.colums)
x_scaled.head()



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train.head()

#Using Ensemble Techniques

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
rfr=RandomForestRegressor()
gb=GradientBoostingRegressor()
ad=AdaBoostRegressor()


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

for i in [rfr,gb,ad] :
i.fit(x_train,y_train)
y_pred=i.predict(x_test)
test_score=r2_score(y_test,y_pred)
train_score=r2_score(y_train, i.predict(x_train))
if abs(train_score-test_score)<=0.2:
print(i)


print("R2 score is",r2_score(y_test,y_pred))
print("R2 for train data",r2_score(y_train, i.predict(x_train)))
print("Mean Absolute Error is",mean_absolute_error(y_pred,y_test))
print("Mean Squared Error is",mean_squared_error(y_pred,y_test))
print("Root Mean Squared Error is", (mean_squared_error(y_pred,y_test,squared=False)))


#Regression Model

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

knn=KNeighborsRegresso()
svr=SVR()
dt=DecisionTreregressor

for i in [knn,svr,dt]:
i.fit(x_train,y_train)
y_pred=i.predict(x_test)
test_score=r2_score(y_test,y_pred)
train_score=r2_score(y_train,i.predict(x_train))
if abs(train_score-test_score)<=0.1:
print(i)

print("R2 score is",r2_score(y_test,y_pred))
print("R2 for train data",r2_score(y_train,i.predict(x_train)))
print("Mean Absolute Error is",mean_absolute_error(y_pred,y_test))
print("Mean Squared Error is",mean_squared_error(y_pred,y_test))
print("Root Mean Squared Error is", (mean_squared_error(y_pred,y_test,squared=False)))


##cheaking coss validation for randomforestregessor


from sklearrn.model_selection import cross_val_score
for i in range(2,5):
  cv=cross_val_score(rfr,x,y,cv=i)
  print(rfr,cv.mean())


#HYPETUNING THE MODEL

from sklearn.model_selection import RandomizedSearchCV

param_grid={'n_estimation':[10,30,50,70,100],'max_depth':[None,1,2,3],'max_features':['auto','sqrt']}
rfr=RandomForesRegressor()
rf_res=RandomizedsearchCV(estimator=rfr,param_distbutions=param_grid,cv=3,verbose=2,n_jobs=-1)

rf_res.fit(x_train,y_train)




gb=GradientBoostingRegressor()
gb_res=RandomizedSearchCV(estimator=gb,param_distributions=param_grid,cv=3,verbose=2,n_job=-1)

gb_res.fit(x_train,y_train)




#ACCURACY

rfr=RandomForestRegressor(n_estimator=10,max_features='sprt',max_depth=None)
rfr.fit(x_train,y_train)
y_train_pred=rfr.predict(x_train)
y_test_pred=rfr.predict(x_test)
print("train accuracy",r2_score(y_train_pred,y_train))
print("test accuracy",r2_score(y_test_pred,y_test))

knn=KNeighborsRegressor(n_neighbors=2,algorithm='auto',metric_params=None,n_jobs=-1)
knn.fit(x_train,y_train)
y_train_pred=knn.predict(x_train)
y_test_pred=knn.predict(x_test)
print("train accuracy",r2_score(y_train_pred,y_train))
print("test accuracy",r2_score(y_test_pred,y_test))

#EVALUATION PERFORMANCE OF THE MODEL AND SAVING THE MODEL

rfr=RandomForestRegressor(n_estimator=10,max_features='sprt',max_depth=None)
rfr.fit(x_train,y_train)
y_train_pred=rfr.predict(x_train)
y_test_pred=rfr.predict(x_test)
print("train accuracy",r2_score(y_train_pred,y_train))
print("test accuracy",r2_score(y_test_pred,y_test))

price_list=pd.DataFrame({'price':price})
price_list


import pickle
pickle.dump(rfr,open('model1.pkl','wb'))







 
