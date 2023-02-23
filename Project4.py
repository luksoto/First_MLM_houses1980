#%%
#Import this code just one time, comment after that.  
#!{sys.executable} -m pip install seaborn scikit-learn

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
