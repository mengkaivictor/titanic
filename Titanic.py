
# coding: utf-8

# In[288]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[289]:


data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")


# In[290]:


data_train.head(5)


# In[291]:


data_train.info()


# In[292]:


data_test.info()


# In[293]:


def drop_feas(df):
    drop_feas = ["Name", "Ticket"]
    df = df.drop(drop_feas, axis=1)
    return df

def simplify_age(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 100)
    group_names = ["No_info", "Baby", "Child", "Teenager", "Student", "Young Adult", "Adult", "Senior"]
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def fill_na_fare(df):
    df.Fare = df.Fare.fillna(method="bfill")
    return df
    
def simplify_cabin(df):
    df.Cabin = df.Cabin.fillna("N")
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def fill_na_embarked(df):
    df.Embarked = df.Embarked.fillna("S")
    return df

def transform_data(df):
    df = simplify_age(df)
    df = fill_na_fare(df)
    df = simplify_cabin(df)
    df = fill_na_embarked(df)
    df = drop_feas(df)
    return df


# In[294]:


df_train = transform_data(data_train)
df_test = transform_data(data_test)


# In[295]:


df_train.sample()


# In[296]:


df_train.info()


# In[297]:


df_test.info()


# In[298]:


df_train.sample()


# In[299]:


df_test.sample()


# In[300]:


from sklearn import preprocessing


# In[301]:


def encode_feas(df_train, df_test):
    feas = ["Sex", "Age", "Cabin", "Embarked"]
    df_combined = pd.concat([df_train[feas], df_test[feas]])
    
    for fea in feas:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[fea])
        df_train[fea] = le.transform(df_train[fea])
        df_test[fea] = le.transform(df_test[fea])
        
    return df_train, df_test


# In[302]:


df_train, df_test = encode_feas(df_train, df_test)


# In[303]:


df_train.sample()


# In[304]:


df_test.sample()


# In[305]:


df_train = df_train.set_index(df_train.PassengerId).drop(["PassengerId"], axis=1)
df_test = df_test.set_index(df_test.PassengerId).drop(["PassengerId"], axis=1)


# In[306]:


df_train.sample()


# In[307]:


df_test.sample()


# In[308]:


y = df_train.Survived
X = df_train.drop(["Survived"], axis=1)


# In[270]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# In[276]:


clf = RandomForestClassifier()
grid_params = {"n_estimators": [4,6,8,10,12],
              "criterion": ["gini", "entropy"],
              "max_depth": [4,6,8,10],
              "max_features": ["auto", "sqrt", "log2"]}
model = GridSearchCV(clf, grid_params, "accuracy")
model = model.fit(X,y)


# In[277]:


model.best_params_


# In[278]:


best_model = model.best_estimator_
best_model.fit(X,y)
predictions = best_model.predict(df_test)


# In[284]:


output = pd.DataFrame({"PassengerId": df_test.index, "Survived": predictions})


# In[287]:


output.to_csv("submission.csv")


# In[39]:


sns.barplot(x="Pclass", y="Survived", hue="Sex", data=data_train)


# In[24]:


sns.pointplot(x="Fare", y="Survived", hue="Sex", data=data_train)

