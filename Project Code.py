
# coding: utf-8

# In[ ]:


# import all required files
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, confusion_matrix, classification_report
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge, LinearRegression, BayesianRidge
from sklearn.preprocessing import StandardScaler, Imputer, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, RandomForestRegressor
import itertools


# In[ ]:


# import data
df = pd.read_csv('dataset2.csv',sep=',')


# In[ ]:


# create dataframe with selected columns
columns = ['Make','Year','Sales','annual fuel consumption',
           'T to charge120','T to charge240','city epm100' ,
           'co2','combinedCD GPM100','cylinders','displ','drive','fuelCost','fuelType',
           'elec compumption kw100m','LV','PV','trany','VClass','youSaveSpend',
           'charge240b','mpg City total',
           'mpg Hwy total','mpg Comb total','MSRP Low']

df_2 = df[columns]
df_2 = df_2.dropna(subset=['annual fuel consumption'])
df_2=pd.get_dummies(df_2)
df_2.head()


# In[ ]:


# define predictor and response variable
df_2 = df_2.dropna(axis=0)
X = df_2.drop(['Sales'],axis=1).values #sets x and converts to an array
y = df_2['Sales'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# In[ ]:


# Method used for Hyper parameter tuning
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

def optimizer(model, X_train, y_train):
    fold_accuracies = cross_val_score(model, X_train, y_train, cv=kfold)
    print("Cross-validation score:\n{}".format(fold_accuracies))
    print("Average cross-validation score: {:.2f}".format(fold_accuracies.mean()))
    
    return fold_accuracies.mean()


# In[ ]:


#Experiment 1 : DecisionTreeRegressor

accuracy_dt = []
def crossValidationForDecisionTreeRegressor():
    for v in [None, 5, 10, 15, 20, 25, 30, 35, 40, 45,50, 55, 60, 70]:
        dt = DecisionTreeRegressor(max_depth=v)
        print("For DecisionTreeRegressor max depth ", v)
        accuracy_dt.append(optimizer(dt, X_train, y_train))
        return dt;
    
dt = crossValidationForDecisionTreeRegressor()
components = [None, 5, 10, 15, 20, 25, 30, 35, 40, 45,50, 55, 60, 70];
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(components,accuracy_dt, "b-")
plt.xlabel('Max Depth')
plt.ylabel('accuracy')
plt.title('DecisionTreeRegressor')


# In[ ]:


#Experiment 2 : RandomForestRegressor

accuracy_rf=[]
def crossValidationForRandomForestRegressor():
    for v in [None,5, 10, 15, 20, 25, 30, 35, 40, 45,50, 55, 60, 70]:
        rf = RandomForestRegressor(max_depth=v)
        print("For RandomForestRegressor max depth ", v)
        accuracy_rf.append(optimizer(rf, X_train, y_train))
        return rf;
    
rf = crossValidationForRandomForestRegressor()

components = [None, 5, 10, 15, 20, 25, 30, 35, 40, 45,50, 55, 60, 70];
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(components,accuracy_rf, "b-")
plt.xlabel('Max Depth')
plt.ylabel('accuracy')
plt.title('RandomForestRegressor')


# In[ ]:


#Experiment 3 : GradientBoostingRegressor

accuracy_gb=[]
def crossValidationForGradientBoostingRegressor():
    for v in [None, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        bc = GradientBoostingRegressor(max_depth=v)
        print("For GradientBoostingRegressor maxdepth ", v)
        accuracy_gb.append(optimizer(bc, X_train, y_train))
        return bc;

bc = crossValidationForGradientBoostingRegressor()

components = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9];
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(components,accuracy_gb, "b-")
plt.xlabel('Max Depth')
plt.ylabel('accuracy')
plt.title('GradientBoostingRegressor')


# In[ ]:


accuracy_gb=[]
def crossValidationForGradientBoostingRegressor():
    for v in [10, 15, 20, 25, 30, 35, 40, 45,50, 55, 60, 70]:
        bc = GradientBoostingRegressor(max_depth=v)
        print("For GradientBoostingRegressor maxdepth ", v)
        accuracy_gb.append(optimizer(bc, X_train, y_train))
        return bc;

bc = crossValidationForGradientBoostingRegressor()

components = [10, 15, 20, 25, 30, 35, 40, 45,50, 55, 60, 70];
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(components,accuracy_gb, "b-")
plt.xlabel('Max Depth')
plt.ylabel('accuracy')
plt.title('GradientBoostingRegressor')


# In[ ]:


#Experiment 4 a : BaggingRegressor

accuracy_bg=[]
def crossValidationForBaggingRegressor():
    for v in [ 10, 20, 30, 40, 50, 55, 70, 80, 90, 100]:
        bc = BaggingRegressor(n_estimators=v)
        print("For BaggingRegressor n_estimators ", v)
        accuracy_bg.append(optimizer(bc, X_train, y_train))
        return bg;
    
bg = crossValidationForBaggingRegressor()
components = [ 10, 20, 30, 40, 50, 55, 70, 80, 90, 100];
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(components,accuracy_bg, "b-")
plt.xlabel('Max Depth')
plt.ylabel('accuracy')
plt.title('GradientBoostingRegressor')


# In[ ]:


#Experiment 4 b : BaggingRegressor with RandomForestRegressor
accuracy_bgrf=[]
def crossValidationForBaggingRegressor():
    for v in [None,5, 10, 15, 20, 25, 30, 35, 40, 45,50, 55, 60, 70]:
        rf = RandomForestRegressor(max_depth=v)
        bgrf = BaggingRegressor(rf)
        print("For BaggingRegressor with RandomForestRegressor max_depth ", v)
        accuracy_bgrf.append(optimizer(bgrf, X_train, y_train))
        return bgrf;
bgrf = crossValidationForBaggingRegressor()

components = [None,5, 10, 15, 20, 25, 30, 35, 40, 45,50, 55, 60, 70];
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(components,accuracy_bgrf, "b-")
plt.xlabel('Max Depth')
plt.ylabel('accuracy')
plt.title('BaggingRegressor with RandomForestRegressor')


# In[ ]:


#Experiment 5 : AdaBoostRegressor with RandomForestRegressor

accuracy_abcrf=[]
def crossValidationForAdaBoostRegressor():
    
    for v in [None,5, 10, 15, 20, 25, 30, 35, 40, 45,50, 55, 60, 70]:
        rf = RandomForestRegressor(max_depth=v)
        abc = AdaBoostRegressor(rf)
        print("For AdaBoostRegressor max_depth ", v)
        accuracy_bgrf.append(optimizer(abc, X_train, y_train))
        return abc;
abc = crossValidationForAdaBoostRegressor()

components = [None,5, 10, 15, 20, 25, 30, 35, 40, 45,50, 55, 60, 70];
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(components,accuracy_abcrf, "b-")
plt.xlabel('Max Depth')
plt.ylabel('accuracy')
plt.title('AdaBoostRegressor with RandomForestRegressor')


# In[ ]:


# Feature Selection

def fit_model(X, y):
    bc = GradientBoostingRegressor(max_depth=8, random_state=42)
    bc.fit(X,y)
    RSS = mean_squared_error(y,bc.predict(X)) * len(y)
    R_squared = bc.score(X,y)
    return RSS, R_squared


RSS_list=[]
R_squared_list=[]
feature_list=[]
numb_features=[]

for k in range(1,len(df_2.columns)-1):
    for combo in itertools.combinations(range(0,len(df_2.columns)-1),k):
        tmp_result = fit_model(X_train[:, list(combo)],y_train)
        RSS_list.append(tmp_result[0])
        R_squared_list.append(tmp_result[1])
        feature_list.append(combo)
        numb_features.append(len(combo))   
        print({'numb_features': len(combo),'RSS': tmp_result[0], 'R_squared':tmp_result[1],'features':df_2.columns[list(combo)]})

f_df = pd.DataFrame({'numb_features': numb_features,'RSS': RSS_list, 'R_squared':R_squared_list,'features':feature_list})


# In[ ]:


# Testing 1

model_gb = GradientBoostingRegressor(max_depth=8, random_state=42)
model_gb.fit(X_train, y_train)
y_predicted_gb = model_gb.predict(X_test)


# mesurement metrics

# 1
print("mean_absolute_error:", mean_absolute_error(y_test, y_predicted_gb))

# 2
print("r2_score:",r2_score(y_test, y_predicted_gb))

#3
print("mean_squared_error:", mean_squared_error(y_test, y_predicted_gb))

# 4
print("err", y_predicted_gb-y_test)


# In[ ]:


# Testing 2

model_rf = RandomForestRegressor(max_depth=15)
model_abr = AdaBoostRegressor(rf)

model_abr.fit(X_train, y_train)
y_predicted_abr = model_abr.predict(X_test)


# mesurement metrics

# 1
print("mean_absolute_error:", mean_absolute_error(y_test, y_predicted_abr))

# 2
print("r2_score:",r2_score(y_test, y_predicted_abr))

#3
print("mean_squared_error:", mean_squared_error(y_test, y_predicted_abr))

# 4
print("err", y_predicted_abr-y_test)

