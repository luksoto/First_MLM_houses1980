#  Client Report - [Project 4 Machine Learning]
  
__Lucas Soto__
  
##  Elevator pitch
  
  
The main idea behind this project was to create a model that is able to predict when a house was built before 1980. For this reason, I used the data from the state of Colorado and chose some variables from the data as features. Also, I used the Tree Decision Classifier model for this project. 
  
###  GRAND QUESTION 1
  
####  Create 2-3 charts that evaluate potential relationships between the home variables and before1980.
  
In a way to look for variables that show any realtionship before 1980, I decided to used the livearea variable. This variable show the square footage that is liveable in the house. As result I found that there is not relevant information or relatioship between the variable for houses before 1980. We can observe that the distribution is very similar for all the years with exeption after 2000 there are some outliers. 
In my second graph I chose the selling price variable, but I found that this variable is not very relevant for my relatioship that I am looking for. We can observe that the before 2000 most of the houses prices are below 2 million dollares. 
  
#####  TECHNICAL DETAILS
  
  
```python
## First graph
graph=(alt.Chart(dwellings_ml).encode(
    x=alt.X('yrbuilt', scale=alt.Scale(zero=False),axis=alt.Axis(title='Year the House was Built')),
    y=alt.X('livearea', axis=alt.Axis(title='Square Footage that is Liveable'))
  
).mark_point().properties(title="Living Area as an Indicator for Houses Before 1980")
)
  
graph.save('graph1.png')
  
## Second graph
graph2=(alt.Chart(dwellings_ml).encode(
    x=alt.X('yrbuilt',scale=alt.Scale(zero=False),axis=alt.Axis(title="Year the House was Built")),
    y=alt.X('netprice', axis=alt.Axis(title="House Prices"))
).mark_bar().properties(title=" House Price as an Indicator for Houses Before 1980")
)
graph2.save("graph2.png")
```
  
![](graph1.png )
  
![](graph2.png )
  
  
  
###  GRAND QUESTION 2
  
####  Can you build a classification model (before or after 1980) that has at least 90% accuracy for the state of Colorado to use (explain your model choice and which models you tried)?
  
  
As I mentioned before I used the Tree Decision Classifier, since I found that it fits better with the data I am working and with the tipe of prediction I was working. For example, I wanted the model to classify  as before 1980 or after 1980 the houses according with some specif characteristics. For this reason, since I wanted classify the tree decision classifier will be one of my options. Another reasons because I choose this model is because It is easy to interpreter and visualize.
I tried other models as GradientBoostingClassifier but I could not obtain an accuracy score bigger than 0.80. For this reason, I decided to switch and add some attirbutes to the tree decision classifier model as the criterion="entropy" that improved the accuracy score to 90%. 
  
#####  TECHNICAL DETAILS
  
  
```python
x=dwellings_ml.drop(['yrbuilt','parcel', 'before1980'],axis=1)
y=dwellings_ml.filter(['before1980' ], axis=1)
  
#%%
X_train, X_test, y_train, y_test= train_test_split(
    x,
    y,
    test_size=0.34,
    random_state= 76
)
#%%
clf=tree.DecisionTreeClassifier(criterion="entropy", random_state= 0)
clf.fit(X_train,y_train)
y_predictions=clf.predict(X_test)
#%%
##looking at the accuraency of the test
metrics.accuracy_score(y_test, y_predictions)
  
```
###  GRAND QUESTION 3
  
####  Will you justify your classification model by detailing the most important features in your model (a chart and a description are a must)?
  
The bar chart below shows how each feature incluenfe in the classifier model, and in the first place we have the arcstyle variable with more than 24%. It made a lot of sense since the architechture style change a long of the time, then it should be a variable that will differenciate houses before 1980. In the same way other features influence in the model but not with the same proportion. Something that caught my attention was the livearea has more influence in the model than the selling price. These two variables I graphed for question 1 but did not show relevant information.  
  
#####  TECHNICAL DETAILS
  
  
```python
df_features = pd.DataFrame(
    {'f_names': X_train.columns,
    'f_values': clf.feature_importances_}).sort_values('f_values', ascending = False)
  
df_features_top= df_features.query('f_values > 0.02')
df_features_top
  
variablesChart = alt.Chart(df_features_top).mark_bar(color='black').encode(
    x=alt.X('f_values', axis=alt.Axis(title='Percentage Each Feature Affect the Model')),
    y=alt.Y('f_names', axis= alt.Axis(title='Feature Name'),sort='-x')
).properties(
    title='How Each Feature Affect the Model ?'
  
)
variablesChart.save("variable.png")
  
```
  
![](variable.png )
  
  
  
###  GRAND QUESTION 4
  
####  Can you describe the quality of your classification model using 2-3 evaluation metrics? You need to provide an interpretation of each evaluation metric when you provide the value.
  
The first and simple way to evaluate the performance of the method is Accuracy. This is the total amount of correct predictions divided by the total amount of data points. In this case the accuracy is 0.90.
Another useful tool we have is Precision. This is ability of the model to indetify the  only the relevant data points. It is calculate by dividing the number of true positivies by the sum of the number of true positives with the number of false positives. In this case the precision is 0.93.   
  
#####  TECHNICAL DETAILS
  
  
```python
print(classification_report(y_test,y_predictions))
  
```
  
![](report.png )
  
  
  
  
  
  
  
  
  
  
  
  
  
##  APPENDIX A (PYTHON CODE)
  
```python
  
#%%
import pandas as pd 
import numpy as np
import seaborn as sns
import altair as alt
#%%
# the from imports
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
alt.data_transformers.enable('json')
#%%
  
#reading and importing the data. 
  
dwellings_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
  
dwellings_ml.head(5)
  
#%%
  
graph=(alt.Chart(dwellings_ml).encode(
    x=alt.X('yrbuilt', scale=alt.Scale(zero=False),axis=alt.Axis(title='Year the House was Built')),
    y=alt.X('livearea', axis=alt.Axis(title='Square Footage that is Liveable'))
  
).mark_point().properties(title="Living Area as an Indicator for Houses Before 1980")
)
  
graph.save('graph1.png')
# %%
graph2=(alt.Chart(dwellings_ml).encode(
    x=alt.X('yrbuilt',scale=alt.Scale(zero=False),axis=alt.Axis(title="Year the House was Built")),
    y=alt.X('netprice', axis=alt.Axis(title="House Prices"))
).mark_bar().properties(title=" House Price as an Indicator for Houses Before 1980")
)
graph2.save("graph2.png")
  
## building a clasification model
#%%
## set the variables
x=dwellings_ml.drop(['yrbuilt','parcel', 'before1980'],axis=1)
y=dwellings_ml.filter(['before1980' ], axis=1)
  
#%%
X_train, X_test, y_train, y_test= train_test_split(
    x,
    y,
    test_size=0.34,
    random_state= 76
  
)
  
#%%
clf=tree.DecisionTreeClassifier(criterion="entropy", random_state= 0)
clf.fit(X_train,y_train)
y_predictions=clf.predict(X_test)
#%%
##looking at the accuraency of the test
metrics.accuracy_score(y_test, y_predictions)
#%%
#ANSWER QUESTION 3
##this shows what variables have most impact in the test. 
df_features = pd.DataFrame(
    {'f_names': X_train.columns,
    'f_values': clf.feature_importances_}).sort_values('f_values', ascending = False)
  
df_features_top= df_features.query('f_values > 0.02')
df_features_top
  
#%%
## now creates a graph to show how each variable affected the test. 
##WHY ALL MY GRAPHS DONT WORK ?
  
variablesChart = alt.Chart(df_features_top).mark_bar(color='black').encode(
    x=alt.X('f_values', axis=alt.Axis(title='Percentage Each Feature Affect the Model')),
    y=alt.Y('f_names', axis= alt.Axis(title='Feature Name'),sort='-x')
).properties(
    title='How Each Feature Affect the Model ?'
  
)
variablesChart.save("variable.png")
#%%
metrics.plot_roc_curve(clf, X_test, y_test)
  
#%%
#ASWER QUESTION 4
print(classification_report(y_test,y_predictions))
  
#%%
  
```
  