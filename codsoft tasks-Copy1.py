#!/usr/bin/env python
# coding: utf-8

# In[25]:


#TASK 1 TITANIC SURVIVAL prediction
# Importing necessary libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Loading the Titanic dataset

titanic_df = pd.read_csv('C:\\Users\\511ws\\Downloads\\Titanic.csv')

# Data preprocessing
# Dropping unnecessary columns and handle missing values
titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)


label_encoders = {}
for col in ['Sex', 'Embarked']:
    label_encoders[col] = LabelEncoder()
    titanic_df[col] = label_encoders[col].fit_transform(titanic_df[col])

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(titanic_df.drop('Survived', axis=1), titanic_df['Survived'], test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)  # Using RandomForestClassifier for model training
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_val, y_pred))


# In[53]:


# TASK 3 : IRIS FLOWER CLASSIFICATION

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
iris_df = pd.read_csv('C:\\Users\\511ws\\Downloads\\IRIS.csv')

# Prepare the data
X = iris_df.drop('species', axis=1)
y = iris_df['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[56]:


# TASK 4 : Sales Prediction Using Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('C:\\Users\\511ws\\Downloads\\advertising.csv')

# Prepare the data
X = data[['TV', 'Radio', 'Newspaper']]  # Features
y = data['Sales']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make sales predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Display actual and predicted sales values
sales_comparison = pd.DataFrame({'Actual Sales': y_test, 'Predicted Sales': y_pred})
print(sales_comparison)


# In[ ]:




