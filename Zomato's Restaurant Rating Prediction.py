# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:10:52 2019

@author: Acer
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r"D:\projects\zomato.csv")

data.describe()
data.columns

# Transforming rate column


data['rate_new'] = data['rate'].astype(str)
data['rate_new'] = data['rate_new'].apply(lambda x: x.split('/')[0])# Dealing with instanced with 'NEW'
data['rate_new'] = data['rate_new'].apply(lambda x: x.replace('NEW', str(np.nan)))
data['rate_new'] = data['rate_new'].apply(lambda x: x.replace('-', str(np.nan)))
# Changing data type
data['rate_new'] = data['rate_new'].astype(float)

data.drop(['rate'], axis=1, inplace=True)
print(f'{type(data["rate_new"][0])}')

data['approx_cost(for two people)'] = data['approx_cost(for two people)'].str.replace(',','').apply(lambda x:float(x))

#%%
#Dropping unnecessary data
data.drop(['url', 'address', 'dish_liked', 'phone', 'reviews_list', 'menu_item','location'], axis=1, inplace=True)

# Looking for null data
data.isnull().sum()
data = data.dropna(subset=['rate_new', 'approx_cost(for two people)'])
data = data.fillna('Not defined')
data.isnull().sum()

data.reset_index(drop=True)
data.describe()
data
#===========================EDA===========================
#%%
#1.Restaurant Rate Distribution
data['rate_new'].describe()

sns.set(style='darkgrid',palette='muted',color_codes=True)
fig, ax=plt.subplots(figsize=(12,5))
sns.distplot(data['rate_new'],bins=30,color='blue')
ax.set_title('Restaurant Rate Distribution',size=13)
ax.set_xlabel('Rate')
plt.show()
#%%
#2. Approx. cost of 2 people

data['approx_cost(for two people)']
sns.set(style='darkgrid',palette='muted',color_codes=True)
fig, ax=plt.subplots(figsize=(12,5))
sns.distplot(data['approx_cost(for two people)'],bins=10,color='blue')
ax.set_title('Approx cost for two people')
ax.set_xlabel('cost')
plt.show()
#%%
#3.Finding Outliers
#Online_Order
fig, ax=plt.subplots(figsize=(12,7))
sns.boxplot(x='online_order',y='rate_new',data=data)

#BookTable
fig, ax=plt.subplots(figsize=(12,7))
sns.boxplot(x='book_table',y='rate_new',data=data)

#%%
#4.Correlation between rating and cost
fig, ax = plt.subplots(figsize=(10 , 10))
sns.scatterplot(x='rate_new', y='approx_cost(for two people)', data=data, ax=ax)
ax.set_title('Correlation Between Rate and Approx Cost', size=14)
plt.show()


#%%
#5.Correlation between rating and Online Order or Booking Tables
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
sns.scatterplot(x='rate_new', y='approx_cost(for two people)', hue='online_order', 
                data=data, ax=axs[0], palette=['navy', 'crimson'])
sns.scatterplot(x='rate_new', y='approx_cost(for two people)', hue='book_table', 
                data=data, ax=axs[1], palette=['navy', 'crimson'])
axs[0].set_title('Cost and Rate Distribution by Online Order', size=14)
axs[1].set_title('Cost and Rate Distribution by Book Table', size=14)
plt.show()
#%%
data.groupby(by='online_order').mean()
#%%
data.groupby(by='book_table').mean()
#%%
#6.Top Rated Restaurant
grouped_rate = data.groupby(by='name', as_index=False).mean()
top_rating = grouped_rate.sort_values(by='rate_new', ascending=False).iloc[:10, np.r_[0, -1]]
top_rating

top_rating.iloc[1, 0] = 'Santa Spa Cuisine'


# Plotting
fig, ax = plt.subplots(figsize=(13, 5))
ax = sns.barplot(y='name', x='rate_new', data=top_rating, palette='Blues_d')
ax.set_xlim([4.7, 4.95])
ax.set_xlabel('Mean Rate')
ax.set_ylabel('')
for p in ax.patches:
    width = p.get_width()
    ax.text(width+0.007, p.get_y() + p.get_height() / 2. + 0.2, '{:1.2f}'.format(width), 
            ha="center", color='grey')

ax.set_title('Top 10 Restaurants in Bengaluru by Rate', size=14)
plt.show()
#%%
#Label Encoding


from sklearn.preprocessing import LabelEncoder

lb_en=LabelEncoder()
data['online_order']=lb_en.fit_transform(data['online_order'])
data['online_order'].unique()
data['online_order']
#%%
data['book_table']=lb_en.fit_transform(data['book_table'])
data['book_table'].unique()
data['book_table']
#%%
data['listed_in(type)']=lb_en.fit_transform(data['listed_in(type)'])
data['listed_in(type)']
#%%
data['listed_in(city)'].unique()
data['listed_in(city)']=lb_en.fit_transform(data['listed_in(city)'])
data['listed_in(city)']
#%%
data=pd.read_excel(r"D:\projects\zomato2.xlsx")
data['rest_type'].unique()
data['rest_type']=lb_en.fit_transform(data['rest_type'])
data['rest_type'].unique()
data['rest_type']


data.drop(['cuisines'], axis=1, inplace=True)
data.columns
#data['rate_new'].unique()
data['listed_in(type)'].unique()
#%%
corr = data.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)

#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

data.columns
x=data.iloc[:,1:8]
y=data['rate_new']

sc=StandardScaler()
sc.fit(x)
x=sc.transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=20)
#kfold=StratifiedKFold(n_splits=10,random_state=48)
#%%
#LASSO
from sklearn.linear_model import Lasso
from sklearn.metrics import  mean_squared_error
from sklearn.metrics import r2_score
alpha=0.1
lasso=Lasso(alpha=alpha)
lasso.fit(x_train,y_train)
y_train_pred=lasso.predict(x_train)
y_test_pred=lasso.predict(x_test)
print(lasso.coef_)
print('MSE train: %.3f,test: %.3f' % (mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
print('R^2 train: %.3f,test: %.3f' % (r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))

r2_score_lasso=r2_score(y_test,y_test_pred)
print(lasso)
print("r^2 on test data: %f" % r2_score_lasso)

predictors=data.columns.values[1:8]
coef=pd.Series(lasso.coef_,predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients')
#%%
#RandomForest

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestRegressor(n_estimators = 100))
sel.fit(x_train, y_train)
sel.get_support()
selected_feat= x_train.columns[(sel.get_support())]
print(selected_feat)

#%%
from xgboost import XGBRegressor
from numpy import sort

model = XGBRegressor()
model.fit(x_train, y_train)
# make predictions for test data and evaluate
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
r2 = r2_score(y_test,y_test_pred)
print("R2: %.2f%%" % (r2 * 100.0))
mse=mean_squared_error(y_test,y_test_pred)
print("MSE: %.2f%%" % (mse * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(x_train)
	# train model
	selection_model = XGBRegressor()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(x_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	r2 = r2_score(y_test,y_test_pred)
    #mse=mean_squared_error(y_test,y_test_pred)
	print("Thresh=%.3f, n=%d, r2: %.2f%%" % (thresh, select_X_train.shape[1], r2*100.0))
    
#%%
#Model Building
#RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

RForest=RandomForestRegressor(n_estimators=5,random_state=329,min_samples_leaf=.0001)
RForest.fit(x_train,y_train)
y_predict=RForest.predict(x_test)
from sklearn.metrics import r2_score
print("Random forest:", r2_score(y_test,y_predict))
#results=cross_val_score(RForest,x_train,y_train,cv=kfold)
#print("CVS:",results)

#Linear Regression
lm=LinearRegression()
lm.fit(x_train,y_train)
y_pred=lm.predict(x_test)
from sklearn.metrics import r2_score
print("Linear Regression:",r2_score(y_test,y_pred))
#results=cross_val_score(lm,x_train,y_train,cv=kfold)
#print("CVS:",results)

#DecisionTree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from os import  system

DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)
from sklearn.metrics import r2_score
print("Decision Tree:",r2_score(y_test,y_predict))

#results=cross_val_score(DTree,x_train,y_train,cv=kfold)
#print("CVS:",results)

#SVM regressor
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train,y_train)
y_predict=regressor.predict(x_test)
from sklearn.metrics import r2_score
print("SVM regressor:", r2_score(y_test,y_predict))

#XGBoost Regressor
import xgboost

xgb = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.5, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=7)
xgb.fit(x_train,y_train)
predictions = xgb.predict(x_test)
print("XGB:",r2_score(y_test,predictions))

#KNN
from sklearn import neighbors
r2_val = [] 
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train, y_train)  #fit the model
    pred=model.predict(x_test) #make prediction on test set
    r2 = r2_score(y_test,pred) #calculate rmse
    r2_val.append(r2) #store rmse values
    print('R2 for k= ' , K , 'is:', r2)
    
    