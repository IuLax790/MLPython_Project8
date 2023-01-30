import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import confusion_matrix,accuracy_score,mean_squared_error,r2_score

Google_Stock = pd.read_csv("C:\Information_Science\GOOGL.csv")

Google_Stock.Close.plot(figsize=(10,10),color='r')
plt.ylabel("{} Prices".format(Google_Stock))
plt.title("{} Price Series".format(Google_Stock))
plt.show()
sns.displot(Google_Stock["Close"])
sns.displot((Google_Stock["High"]))

X = Google_Stock.drop(["Date","Close","Adj Close","Volume"],axis=1)
y = Google_Stock["Adj Close"]
print(X)
print(y)



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print(y_pred)
def calculate_metrics(y_test,y_pred):
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    r2_scors = r2_score(y_test,y_pred)
    print("MSE:-",mse)
    print("RMSE:-",rmse)
    print("R2 score:-",r2_scors)
print(calculate_metrics(y_test,y_pred))

la = Lasso().fit(X_train,y_train)
ri = Ridge().fit(X_train,y_train)
la_p  = la.predict(X_test)
ri_p = ri.predict(X_test)
calculate_metrics(y_test,la_p)
calculate_metrics(y_test,ri_p)

param_grid = {'C':[0.1,1,10,100,100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}
grid = GridSearchCV(SVR(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
svr = SVR(C=100,gamma=0.01,kernel='rbf')
svr.fit(X_train,y_train)
svr_pred = svr.predict(X_test)
print(svr_pred)

#Predicting Volume

Meta_Stock.Close.plot(figsize=(10,10),color='r')
plt.ylabel("{} Prices".format(Meta_Stock))
plt.title("{} Price Series".format(Meta_Stock))
plt.show()
sns.displot(Meta_Stock["Close"])
sns.displot((Meta_Stock["High"]))

X = Meta_Stock.drop(["Date","Close","Adj Close","Volume"],axis=1)
y = Meta_Stock["Volume"]
print(X)
print(y)



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print(y_pred)
def calculate_metrics(y_test,y_pred):
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    r2_scors = r2_score(y_test,y_pred)
    print("MSE:-",mse)
    print("RMSE:-",rmse)
    print("R2 score:-",r2_scors)
print(calculate_metrics(y_test,y_pred))

la = Lasso().fit(X_train,y_train)
ri = Ridge().fit(X_train,y_train)
la_p  = la.predict(X_test)
ri_p = ri.predict(X_test)
calculate_metrics(y_test,la_p)
calculate_metrics(y_test,ri_p)

param_grid = {'C':[0.1,1,10,100,100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}
grid = GridSearchCV(SVR(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
svr = SVR(C=100,gamma=0.01,kernel='rbf')
svr.fit(X_train,y_train)
svr_pred = svr.predict(X_test)
print(svr_pred)
