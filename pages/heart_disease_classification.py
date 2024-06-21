import pandas as pd
import requests
import plotly.express as px
import dash
from dash import html, dcc, register_page  #, callback # If you need callbacks, import it here.
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from dash import callback
import requests
import plotly.express as px
import numpy as np
import datetime as dt
from datetime import date
# import os
from dash import dash_table


#analysis and viz imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#data preparation imports
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

#ML model imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#metric and analysis imports
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.inspection import permutation_importance

#deployment imports
import joblib

heart_disease_data = pd.read_csv("./data/heart.csv")




register_page(
    __name__,
    name='Heart Disease Classification',
    top_nav=True,
    path='/heart_disease_classification'
)


## -------------------------------------------------------------------------------------------------
### App layout

layout = html.Div([
    html.Div([

        dcc.Markdown(
            '''
            # Part 1 - INTRODUCTION
            ''',style={"text-align":"center"}),
            dcc.Markdown(
            '''                 
            Heart disease describes a variety of related conditions such as Coronary Artery Disease, Acute coronary syndrome, Angina, and Aortic Anuerism. Heart disease can lead to numerous detrimental or fatal conditions such as diabetes, heart failure, and heart attack.

            It is the leading cause of death in the Unites States, according to the CDC. Roughly 660,000 - 700,00 people in the US die each year from heart disease, which is about one-quarter of all yearly deaths. Approximately 18 million people wordwide die due to heart disease annually, which is about 32% of all deaths.

            Given the disease's prevelence and series symptoms, doctors are collecting data from patients to assess a patient's risk for heart disease and prevent it if possible. In this analysis, I will use a set of patient data and use it to predict a patient's risk for having heart disease.

            To do this, I will clean data and then utilize different machine learning models to predict a patient's risk of having heart disease. I will then take the most successful model and deploy it so others could, in theory, use it to predict this risk in real time.

            ### ---- Results ----

            After running several different models on the data, I determined that a random forest model was best able to predict heart disease risk with an accuracy of 96.6% and F1 score of 96.8%. With this finding, I then deployed the model to Heroku so others could (theoretically) use it for prediction purposes.
            '''
            ,dangerously_allow_html=True),
            dcc.Markdown(
            '''
### ---- Data Source and Notes ----

Data source: https://www.kaggle.com/johnsmith88/heart-disease-dataset/version/2

As noted by the original data uploader above: "this data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease."

Data column dictionary:

1. Age
2. Sex
3. Chest pain type (4 values)
4. Resting blood pressure
5. Serum cholestoral in mg/dl
6. Fasting blood sugar > 120 mg/dl
7. Resting electrocardiographic results (values 0,1,2)
8. Maximum heart rate achieved
9. Exercise induced angina
10. Oldpeak = ST depression induced by exercise relative to rest
11. The slope of the peak exercise ST segment
12. Number of major vessels (0-3) colored by flourosopy
13. Thal: 0 = normal; 1 = fixed defect; 2 = reversable defect

### ---- Import Libraries for Project ----
            '''
            ),
            dcc.Markdown(
            '''
```python
#analysis and viz imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#data preparation imports
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

#ML model imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#metric and analysis imports
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.inspection import permutation_importance

#deployment imports
import joblib
```
            '''
            ),
            dcc.Markdown(
'''
## Part 2 - Data Preparation And Exploration

### ---- Load Data ----

```python
heart_disease_data = pd.read_csv("./heart.csv")

#use the head method to get a glimpse of the data structure before moving forward
heart_disease_data.head()
```
'''

            ),
            dash_table.DataTable(heart_disease_data.head().to_dict('records'),
                                 style_table={'overflowX': 'auto'},
                                 style_cell={'minWidth': '90px', 'width': '90px', 'maxWidth': '90px','whiteSpace': 'normal'})
    
    
    
    
    
    
    ]) 
],className="article"
)
