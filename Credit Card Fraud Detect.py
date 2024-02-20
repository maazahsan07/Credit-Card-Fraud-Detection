# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:54:01 2024

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:54:47 2024

@author: Dell
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report

df = pd.read_csv('creditcard.csv')
df.head(5)

df.info()

df.describe()

df.isna().sum()

df.duplicated().sum()

# drop duplication

df.drop_duplicates(df,inplace=True)

print(df.shape)

# how many cases fraud in the dataset

df["Class"].value_counts().plot(kind= "pie",autopct='%1.2f%%', shadow = True)
plt.show()


X = df.drop('Class', axis =1)

y = df['Class']

from sklearn.model_selection import train_test_split
# split the data train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)





from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(max_iter=500)
model1.fit(X_train, y_train)
print('Accuracy training : {:.3f}'.format(model1.score(X_train, y_train)))
pred1 = model1.predict(X_test)
print('Accuracy testing: {:.3f}'.format(model1.score(X_test, y_test)))
print(classification_report(y_test,pred1))



from sklearn.svm import SVC
model2 = SVC()
model2.fit(X_train, y_train)
print('Accuracy training : {:.3f}'.format(model2.score(X_train, y_train)))
pred2 = model2.predict(X_test)
print('Accuracy testing: {:.3f}'.format(model2.score(X_test, y_test)))
print(classification_report(y_test,pred2))



from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier()
model3.fit(X_train, y_train)
print('Accuracy training : {:.3f}'.format(model3.score(X_train, y_train)))
pred3 = model3.predict(X_test)
print('Accuracy testing: {:.3f}'.format(model3.score(X_test, y_test)))
print(classification_report(y_test,pred3))





