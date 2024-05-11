#!/usr/bin/env python
# coding: utf-8

# Author :- Ashutosh Kumar
# Batch :- April
# Domain :- Data Science
# Aim :- To Build a model that Credit Card Fraud Detection.

# In[1]:


import pandas as pd


# In[3]:


data = pd.read_csv('creditcard.csv')


# In[4]:


data.head()


# In[5]:


pd.options.display.max_columns = None


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.shape


# In[9]:


print("Number of columns: {}".format(data.shape[1]))
print("Number of rows: {}".format(data.shape[0]))


# In[10]:


data.info()


# In[11]:


data.isnull().sum()


# In[12]:


from sklearn.preprocessing import StandardScaler


# In[13]:


sc = StandardScaler()
data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))


# In[14]:


data.head()


# In[15]:


data = data.drop(['Time'], axis =1)


# In[16]:


data.head()


# In[17]:


data.duplicated().any()


# In[18]:


data = data.drop_duplicates()


# In[19]:


data.shape


# In[20]:


data['Class'].value_counts()


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[22]:


sns.countplot(data['Class'])
plt.show()


# In[23]:


X = data.drop('Class', axis = 1)
y=data['Class']


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[26]:


import numpy as np


# In[27]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[28]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# In[29]:


classifier = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "AdaBoost Classifier": AdaBoostClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Bagging Classifier": BaggingClassifier(),
    "Extra Trees Classifier": ExtraTreesClassifier(),
    "Stochastic Gradient Descent Classifier": SGDClassifier(),
    "Voting Classifier": VotingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier()),
        ('svc', SVC()),
        ('knn', KNeighborsClassifier())
    ], voting='hard')
}
for name, clf in classifier.items():
    print(f"\n=========={name}===========")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n Accuracy: {accuracy}")
    print(f" Precision: {precision}")
    print(f" Recall: {recall}")
    print(f" F1 Score: {f1}")
    
    # Confusion Matrix
    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
      # Classification Report
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))


# In[30]:


normal = data[data['Class']==0]
fraud = data[data['Class']==1]


# In[31]:


normal.shape


# In[32]:


fraud.shape


# In[33]:


normal_sample = normal.sample(n=473)


# In[34]:


normal_sample.shape


# In[35]:


new_data = pd.concat([normal_sample,fraud], ignore_index=True)


# In[36]:


new_data.head()


# In[37]:


new_data['Class'].value_counts()


# In[38]:


X = new_data.drop('Class', axis = 1)
y= new_data['Class']


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[40]:


classifier = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}

for name, clf in classifier.items():
    print(f"\n=========={name}===========")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n Accuaracy: {accuracy_score(y_test, y_pred)}")
    print(f"\n Precision: {precision_score(y_test, y_pred)}")
    print(f"\n Recall: {recall_score(y_test, y_pred)}")
    print(f"\n F1 Score: {f1_score(y_test, y_pred)}")


# In[41]:


X = data.drop('Class', axis = 1)
y= data['Class']


# In[42]:


X.shape


# In[43]:


y.shape


# In[44]:


from imblearn.over_sampling import SMOTE


# In[45]:


X_res, y_res = SMOTE().fit_resample(X,y)


# In[46]:


y_res.value_counts()


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.2, random_state = 42)


# In[48]:


classifier = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}

for name, clf in classifier.items():
    print(f"\n=========={name}===========")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n Accuaracy: {accuracy_score(y_test, y_pred)}")
    print(f"\n Precision: {precision_score(y_test, y_pred)}")
    print(f"\n Recall: {recall_score(y_test, y_pred)}")
    print(f"\n F1 Score: {f1_score(y_test, y_pred)}")


# In[54]:


dtc = DecisionTreeClassifier()
dtc.fit(X_res, y_res)


# In[55]:


import joblib
joblib.dump(dtc, "credit_card_model.pkl")


# In[56]:


model = joblib.load("credit_card_model.pkl")


# In[57]:


pred = model.predict([[-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62]])


# In[58]:


pred[0]


# In[59]:


if pred[0] == 0:
    print("Normal Transcation")
else:
    print("Fraud Transcation")


# In[ ]:




