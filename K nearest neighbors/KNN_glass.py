import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

#dataset loading
df = pd.read_csv('C:/Users/Tushar Bagul/Desktop/Data_Science/Assignments/KNN/glass.csv')
df.head()
df.describe()
df['Type'].value_counts()

#correlation
plt.figure(figsize = (10,7))
corr = df.corr()
sns.heatmap(corr, annot=True, linewidths=.2)

sns.scatterplot(df['RI'], df['Na'], hue=df['Type'])

plt.figure(figsize = (7,5))
sns.pairplot(df, hue='Type')
plt.show()

#Feature Scaling
scaler = StandardScaler()
scaler.fit(df.drop('Type', axis=1))
scaled_features = scaler.transform(df.drop('Type', axis=1))
scaled_features
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()

#feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = df.iloc[:, 0:9]
y = df.iloc[:, -1]
best_features = SelectKBest(score_func=chi2, k='all')
fit = best_features.fit(X, y)
dfScores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScore = pd.concat([dfcolumns, dfScores], axis=1)
featureScore.columns = ['feature', 'score']
featureScore

#feature importance
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
feat_imp.nlargest(50).plot(kind='barh')
plt.show()

#dropping ca and k
dff = df_feat.drop(['Ca','K'], axis=1)
#Splitting
X_train, X_test, y_train, y_test = train_test_split(
          dff, df['Type'], test_size=0.3, random_state=45)
#using knn
knn = KNeighborsClassifier(n_neighbors=4, metric='manhattan')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred
print(classification_report(y_test, y_pred))
accuracy_score(y_test, y_pred)
























