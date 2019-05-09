
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

data = pd.read_excel('sales revised.xlsx')
data['Model'] = data['Model'].map(str) #converting models to string for matching

category = pd.read_csv('cars(2008-2018).csv')
category['model'] = category['model'].map(str) #converting models to string for matching
sales = data.iloc[:,2:] #removing make and model
row = sales  #iloc[5]
rowt = row.T  #transposing for easier plotting


ind = data['Make'] + ' ' + data['Model'].map(str)

lis = ind.tolist() #making a list of make/model names
rowt.columns = lis #assigning make/model to columns 
g = data[:]

g['category'] = 'nan'
for index, row in g.iterrows():
    try:
        dummy = category[category['make'].str.contains(row['Make'], case = False) & category['model'].str.contains(row['Model'], case = False)]['body_styles'].iloc[0]
    except:
        dummy ='N/A'
    g.at[index,'category'] = dummy

rowt.loc[:,g['category'].str.contains('suv', case = False).values].sum(axis = 1).plot(figsize = [18,18])
plt.show()


# In[72]:


rowt.loc[:,(rowt.sum(axis=0) > 4000000)].plot(figsize = [18,18]) #showing cars that sold more than 4 million total axis = 0 sums column
plt.show()


# In[3]:


chrct = pd.read_excel('vehicles 2.xlsx')


# In[4]:


ev = data[:]

ev['category'] = 'nan'
for index, row in ev.iterrows():
    try:
        dummy = chrct[chrct['make'].str.contains(row['Make'], case = False) & chrct['model'].str.contains(row['Model'], case = False)]['fuelType'].iloc[0]
    except:
        dummy ='N/A'
    ev.at[index,'category'] = dummy
ev


# In[46]:


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
elec = rowt.loc[:,ev['category'].str.contains('Electricity', case = False).values].sum(axis = 1).values
gas =  rowt.loc[:,np.logical_and(ev['category'].str.contains('Gasoline', case = False).values, np.logical_not(ev['category'].str.contains('Electricity', case = False).values))].sum(axis = 1).values
plt.plot(elec)
plt.plot(gas)
ax.set_ylabel('Sales (10x million)')
ax.set_xlabel('Years')
plt.show()


# In[48]:


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
tot = elec + gas
percent = elec/tot*100
plt.plot(percent)
ax.set_ylabel('Percentage of Sales')
ax.set_xlabel('Years')
plt.show()


# In[50]:


np.savetxt("percentage.csv",percent , delimiter=",")


# In[74]:


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
years = np.arange(2011, 2019,1)
logp = np.log(percent*100)
plt.plot(years,logp[11:])
ax.set_ylabel('Log(%Sales)')
ax.set_xlabel('Years')
plt.show()


# In[70]:


from sklearn import linear_model
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
yy = logp[11:]
years = np.arange(2011, 2019,1)
years = years.reshape((-1, 1))
sales  = yy.reshape((-1, 1))
regr = linear_model.LinearRegression()
regr.fit(years, sales)
xtest = np.arange(2011, 2019,1)
xtest = xtest.reshape((-1, 1))
newpred = regr.predict(xtest)
plt.plot(years,sales)
plt.plot(years,newpred)


ax.set_ylabel('Log(%Sales)')
ax.set_xlabel('Years')
plt.show()


# In[82]:


cpi = np.genfromtxt('data3.csv', delimiter=',')


fig, ax1 = plt.subplots(figsize=(12,8))

color = 'tab:red'
ax1.set_xlabel('years')
ax1.set_ylabel('Log Sales',color=color)
ax1.plot(years, sales,color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('CPI Gasoline',color=color)  # we already handled the x-label with ax1
ax2.plot(years, cpi[11:],color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# In[84]:


temp = cpi[11:]
newcpi = temp.reshape((-1, 1))


# In[88]:


fig = plt.figure(figsize=(12,8))

years = np.arange(2011, 2019,1)
years = years.reshape((-1, 1))
params = np.concatenate((years,newcpi), axis = 1)
regr = linear_model.LinearRegression()
regr.fit(params, sales)
# xtest = np.arange(2011, 2019,1)
# xtest = xtest.reshape((-1, 1))
newpred = regr.predict(params)
plt.plot(years,sales,label='Actual')
plt.plot(years,newpred,label='Regression')
fig.legend(loc='center right')
ax.set_ylabel('Log(%Sales)')
ax.set_xlabel('Years')
plt.show()


# In[16]:


temp =ev[ev['category'].str.contains('Electricity', case = False)]
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
   display(temp)


# In[7]:


data.head()


# In[4]:


copy = data[:]
copy.head()


# In[54]:


years = data.iloc[:,2:]
yearlist = years.columns.tolist()

d = []
for index, row in data.iterrows():
    for yr in yearlist:
        d.append((row['Make'], row['Model'], yr, row[yr]))

vehicles = pd.DataFrame(d, columns=('Make', 'Model', 'Year','Sales'))
vehicles.head()


# In[6]:


chrct = pd.read_excel('vehicles 2.xlsx')
chrct


# In[9]:


copy['matches'] = 'nan'
for index, row in copy.iterrows():
    try:
        dummy = chrct[chrct['make'].str.contains(row['Make'], case = False) & chrct['model'].str.contains(row['Model'], case = False)]['year']
        copy.at[index,'matches'] = len(dummy)
    except:
        copy.at[index,'matches'] = 'no match'

copy


# In[10]:


copy.to_csv("sales number of matches with vehicles2.csv", index=False)


# In[55]:


temp = chrct.loc[:, ~chrct.columns.isin(['make', 'model'])]
chrctlist = temp.columns.tolist()
#vehicles = vehicles.iloc[:200,:]


# In[56]:



for ch in chrctlist:
    vehicles[ch] = ''

for index, row in vehicles.iterrows():
    try:
        temp = chrct[chrct['year'] == row['Year']]
        dummy = temp[temp['make'].str.contains(row['Make'], case = False) & temp['model'].str.contains(row['Model'], case = False)  ] 
        for ch in chrctlist: #& chrct['year'] == row['Year']
            vehicles.at[index,ch] = dummy[ch].iloc[0]
    except:
        for ch in chrctlist:
            vehicles.at[index,ch] =''

vehicles


# In[50]:


vehicles.loc[20,'year']


# In[9]:


vehicles.to_csv("revised dataset.csv", index=False)


# In[58]:


for i in range(len(vehicles)):
    if  vehicles.loc[i,'Sales'] > 0 and vehicles.loc[i,'annual fuel consumption'] == '':
        if i !=0:
            if  vehicles.loc[i,'Model'] == vehicles.loc[i-1,'Model'] and vehicles.loc[i-1,'annual fuel consumption'] != '':
                vehicles.iloc[i,4:44] = vehicles.iloc[i-1,4:44]
            elif vehicles.loc[i,'Model'] == vehicles.loc[i+1,'Model']:
                
                if vehicles.loc[i+1,'annual fuel consumption'] == '':
                    vehicles.iloc[i,4:44] = vehicles.iloc[i+2,4:44]
                else:
                    vehicles.iloc[i,4:44] = vehicles.iloc[i+1,4:44]
        elif vehicles.loc[i,'Model'] == vehicles.loc[i+1,'Model']:
             vehicles.iloc[i,4:44] = vehicles.iloc[i+1,4:44]
vehicles


# In[24]:


vehicles.head(50)

