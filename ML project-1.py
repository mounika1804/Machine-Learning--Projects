import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,r2_score
#rmse=mean_squared_error()


train= pd.read_csv(r"C:\\Users\mouni\OneDrive\Desktop\pythonProject\Project - 4\train.csv")
test= pd.read_csv(r"C:\\Users\mouni\OneDrive\Desktop\pythonProject\Project - 4\test.csv")
print(train.head())

print(train.info())
print(train.describe().T)
print(train.corr())
corr=train.corr()
print(corr["SalePrice"])

print(train.corr().abs().nlargest(6,'SalePrice').index)
print(train.corr().abs().nlargest(6,'SalePrice').values[:,37])

sns.jointplot(x="OverallQual",y="SalePrice",data=train,kind="reg");
plt.show()

sns.jointplot(x="GrLivArea",y="SalePrice",data=train,kind="reg");
plt.show()

sns.jointplot(x="GarageCars",y="SalePrice",data=train,kind="reg");
plt.show()

sns.jointplot(x="TotalBsmtSF",y="SalePrice",data=train,kind="reg");
plt.show()

sns.jointplot(x="GarageArea",y="SalePrice",data=train,kind="reg");
plt.show()

lm=LinearRegression()
x=pd.DataFrame(np.c_[train['GrLivArea'],train['OverallQual']],columns=['GrLivArea','OverallQual'])
y=train[["SalePrice"]]







