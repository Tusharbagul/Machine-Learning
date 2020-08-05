import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn import metrics

pd.set_option('display.max_columns', 500)
#reading dataset
df = pd.read_csv('C:/Users/Tushar Bagul/Desktop/Data_Science/Assignments/SVM/forestfires.csv')
df.head()

df.drop(['month', 'day', 'dayfri', 'daymon', 'daysat', 'daysun', 'daythu', 'daytue', 'daywed',
       'monthapr', 'monthaug', 'monthdec', 'monthfeb', 'monthjan', 'monthjul',
       'monthjun', 'monthmar', 'monthmay', 'monthnov', 'monthoct', 'monthsep'], axis='columns', inplace=True)
df.head()
df.isnull().sum()
sns.pairplot(df)
df.columns
df['size_category'].value_counts()

#defining input and target 
X = df.drop(columns=['size_category'])
y = df['size_category']

#EDA
df.corr()
df.hist()
df.plot(kind='density', subplots=True, layout=(4,4))

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=355)

#standard Scaling
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.fit_transform(X_test)

svm = SVC()
svm.fit(X_train_std, y_train)
svm.score(X_train_std, y_train)

y_pred = svm.predict(X_test_std)

print(classification_report(y_test, y_pred))
accuracy_score(y_test, y_pred)

#print test and predicted values
print('True', y_test.values[0:25])
print('Pred', y_pred[0:25])

#Classification Error
print(1 - metrics.accuracy_score(y_test, y_pred))









