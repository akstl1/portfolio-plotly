import dash
from dash import html, dcc, register_page  #, callback # If you need callbacks, import it here.
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import requests
import plotly.express as px
from card import create_card

register_page(
    __name__,
    name='Home',
    top_nav=True,
    path='/'
)

tab1_content = html.Div([
    create_card(
            image="https://github.com/akstl1/portfolio-deployments/blob/main/img/Park2.jpg?raw=true",
            title="Heart Disease Classification",
            description="Some quick example text to build on the card title and",
            position="left",
            url="/home"
            ),
    create_card(
            image="https://github.com/akstl1/portfolio-deployments/blob/main/img/Park2.jpg?raw=true",
            title="Parkinson's Identification",
            description="Some quick example text to build on the card title and",
            position="right",
            url="/home"),
    create_card(
            image="https://github.com/akstl1/portfolio-deployments/blob/main/img/Park2.jpg?raw=true",
            title="Telecom Customer Churn Prediction",
            description="Some quick example text to build on the card title and",
            position="left",
            url="/BI_aggregation")
])

tab2_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 2!", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3",
)


def layout():
    layout = html.Div([
        html.Div([
                    html.Hr(),
                    html.Div([dbc.Tabs([
                                dbc.Tab(tab1_content, label="Python", tab_style={"width":"30%"}),
                                dbc.Tab(tab2_content, label="Power BI", tab_style={"width":"30%"}),
                                dbc.Tab("This tab's content is never seen", label="Tableau", disabled=True, tab_style={"width":"30%"}),
                    ])]),
#                 html.Div([dbc.Card(
#     [
#         dbc.CardImg(src="https://github.com/akstl1/portfolio-deployments/blob/main/img/Park2.jpg?raw=true", top=True),
#         dbc.CardBody(
#             [
#                 html.H4("Card title", className="card-title"),
#                 html.P(
#                     "Some quick example text to build on the card title and "
#                     "make up the bulk of the card's content.",
#                     className="card-text",
#                 ),
#                 dbc.Button("Go somewhere", color="primary"),
#             ]
#         ),
#     ],
#     style={"width": "18rem"},
# )
#                         ])
                

                    ], style={'height':'300px'})
        
    ])
    return layout