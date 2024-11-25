import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
df=pd.read_csv("Book7.csv")
print(df)
df.info()
df.isnull().sum()
df.head()
plt.scatter(df.Built_up_area,df.Rent,color='red',marker='*')
plt.xlabel('Built_up_area')
plt.ylabel('Rent')
plt.title('Scatterplot')
plt.show()
x=df.drop('Rent',axis=1)
print(x)
y=df.Rent
print(y)
from sklearn.model_selection import train_test_split 
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33,random_state=42)
print("xtrain shape : ", xtrain.shape)
print("xtest shape : ", xtest.shape)
print("ytrainshape:",ytrain.shape)
print("ytest shape : ", ytest.shape)
reg=linear_model.LinearRegression()
reg.fit(x,y)
y_pred = reg.predict(xtest)
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)
print(9.7900215*650+1128.879253617697)
train_score=reg.score(xtrain,ytrain)
test_score=reg.score(xtest,ytest)
print('TrainScore(R-Squared):',train_score)
print('Test Score (R-Squared)',test_score)
mse=mean_squared_error(ytest,y_pred)
mae=mean_absolute_error(ytest,y_pred)
print("Mean Square Error : ", mse)
print("Mean Absolute Error : ", mae)
