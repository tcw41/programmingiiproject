####Part 1
##import os
import pandas as pd
import streamlit as st

##The ways I trying to read in a csv file that aren't working in Streamlit

##os.getcwd()

##os.chdir('//Users//taylorwhitelow//Downloads')

##s= pd.read_csv('social_media_usage.csv')
##print(s)

##from pathlib import Path

##s= Path(__file__).parents[1] / 'Downloads/social_media_usage.csv'

##print(s)

##s= pd.read_csv('Downloads/social_media_usage.csv').iloc[1:,:]

s= pd.read_csv('social_media_usage.csv')
##st.dataframe(s)

import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#### Part 2
z= pd.DataFrame ({
   'Column 1':[1,0,0], 
    'Column 2':[0,0,1]
})

print (z)

import numpy as np
def linkedinfuction (x):
    linkedinfuction= np.where(x==1,
                             1,
                             0)
    return (x)

linkedinfuction(z)

#### Part 3
ss= pd.DataFrame({
    'sm_li':np.where(s["web1h"]==1,1,0),
    'income':np.where(s["income"]>9,np.nan, s["income"]),
    'education':np.where(s["educ2"]>8,np.nan, s["educ2"]),
    'parent':np.where(s["par"]==1,1,0),
    'martial':np.where(s["marital"]==1,1,0),
    'female':np.where(s["gender"]==2,1,0),
    'age':np.where(s["age"]<=98,np.nan, s["age"])
    
}).dropna()


#### Part 4 Setting Target Variables
y = ss["sm_li"]
X = ss[["age", "female", "martial", "parent","education","income"]]

#### Part 5 spliting data set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987)

#### Part 6 Logistic regression
lr = LogisticRegression(class_weight='balanced') ##,random_state=0,n_jobs=-1 is the part after this that I took out
lr.fit(X_train, y_train)

#### Part 7 making confusion matrix
y_pred = lr.predict(X_test)
confusion_matrix(y_test, y_pred)

####Part 8 Formatting confusion matrix results
pd.DataFrame(confusion_matrix(y_test, y_pred),
           columns=["Predicted negative", "Predicted positive"],
           index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")

####Part 9 Printing out accruarcy results
print(classification_report(y_test, y_pred))

##Part 10 The Streamlit part

st.header("Predicting if you are a LinkedIn user")

user_age=st.slider("What is your age", min_value=10, max_value=97,value= 18, step=1)
user_gender= st.selectbox("What is your gender? If female select 1, if male select 0",options=[1,0])
user_married= st.selectbox("Are you married? If so select 1, if not select 0",options=[1,0])
user_parent= st.selectbox("Are you a parent? If so select 1, if not select 0",options=[1,0])
st.markdown("What is your education. 1= Less than High School, 2= HS incomplete, 3=HS diploma, 4= some college, 5= two year college, 6= 4 year college, 7= post graduate schooling, 8=PHD")
user_education= st.slider("What is your highest education", min_value=1, max_value=8,value= 1, step=1)
st.markdown("What is your income 1= Less than 10,000 2= 10 to under 20,000 3= 20 to under 30,000 4= 30 to under 40,000 5= 40 to under 50,000 6=50 to under 60,000 7=70 to under 100,000 8=100 to under 150,000 9= more than 150,000")
user_income= st.slider("What is your income", min_value=1, max_value=9,value= 1, step=1)

#User inputs used in logistic regression
person = [user_income,user_education,user_parent,user_married,user_gender,user_age]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

st.subheader('Is this person a Linkedin user?')
if predicted_class > 0:
    label= "This is a Linkedin user!"
else:
    label= "This is not a Linkedin user!"
st.write(label)
    
st.subheader('What is the probaility that this person is a Linkedin user')
st.write(probs)
