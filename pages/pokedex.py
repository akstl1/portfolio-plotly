import dash
from dash import html, dcc, register_page  #, callback # If you need callbacks, import it here.
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import requests
import plotly.express as px

register_page(
    __name__,
    name='Pokedex',
    top_nav=True,
    path='/pokedex'
)



def layout():
    layout = html.Div([
        html.H1(
            [
                "Pokedex"
            ]
        )
        
    ])
    return layout