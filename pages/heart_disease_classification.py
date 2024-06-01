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
import os
from dash import dash_table

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
        
        html.H1("Analyducks"),
        html.H4("A visual analysis of Allan K's rubber duck collection"),
        html.A("Click here to view my portfolio",href= "https://akstl1.github.io/"),
        dcc.Markdown('''
            ```python
            print()
            ```'''
            )
    ],className="title",
    style={
        'text-align': 'center',
        'background-color': 'skyblue',
        'padding-bottom': '5px'
    }
    )
    
])
