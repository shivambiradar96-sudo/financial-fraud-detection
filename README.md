# financial-fraud-detection
Built a financial fraud detection system using Python and RandomForestClassifier. Loaded and cleaned the dataset, prepared features, scaled values, and split data into train/test sets. Evaluated the model through 5-fold cross-validation, classification report, confusion matrix, and feature importance analysis to interpret key fraud indicators
# importing required libraries
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve,auc
import seaborn as sns 
import matplotlib.pyplot as plt


#load csv file & read & check 

df = pd.read_csv(r"file_location")
df.head()

df.info()

#check for null values 
df.isnull().sum()


#split data into features and  target

x = df.drop(['id','Class'],axis=1 , errors='ignore')
y = df['Class']

x.columns.tolist()


#split data intio train and test 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)

x_train.shape

x_test.shape

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


#class distribution 

pd.Series(y_train).value_counts(normalize=True)


#traing the model

rf_model  = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split= 5,
    random_state=42)


# 5 fold  cross validation

cv_score = cross_val_score(rf_model, x_train_scaled, y_train, cv =5, scoring='f1')

#print("\nCross-validation f1 score :", cv_score)

print("avarage f1 score :", np.mean(cv_score))


#fitting the model

rf_model.fit(x_train_scaled, y_train)

#predictions 

y_pred = rf_model.predict(x_test_scaled)

print(classification_report(y_test, y_pred))

#confuysion matrix

plt.figure(figsize=(8,6))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot= True, fmt ='d', cmap='Blues')
plt.title('confusion matrix')

plt.ylabel('true label')
plt.xlabel('predicted label')

plt.show()


#feature importance 
importance = rf_model.feature_importances_
feature_imp = pd.DataFrame({'feature':x.columns,
                            'importance': importance
                            }).sort_values('importance',ascending=False)



feature_imp.head()


#visualise feature importance

plt.figure(figsize=(10,6))
sns.barplot(data=feature_imp,x='importance',y='feature')
plt.title('feature importance ranking')
plt.xlabel('importance score')
plt.tight_layout()
plt.show()


#heat map (correlation matrix)
plt.figure(figsize=(12,8))
correlation_matrix = x.corr()
sns.heatmap(correlation_matrix,cmap = 'coolwarm', center=0, annot=True,ftm ='21')
plt.title('feature correlation matrix')
plt.tight_layout()
plt.show()

