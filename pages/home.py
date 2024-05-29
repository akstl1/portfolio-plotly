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
    name='Home',
    top_nav=True,
    path='/'
)

tab1_content = html.Div([dbc.Card(
    [
        dbc.CardImg(src="https://github.com/akstl1/portfolio-deployments/blob/main/img/Park2.jpg?raw=true", top=True),
        dbc.CardBody(
            [
                html.H4("Parkinson's Identification", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",
                ),
                dbc.Button("Go somewhere", color="primary"),
            ]
        ),
    ],
    style={"width": "18rem","display":"inline-block", "margin":"1rem"},
),dbc.Card(
    [
        dbc.CardImg(src="https://github.com/akstl1/portfolio-deployments/blob/main/img/Park2.jpg?raw=true", top=True),
        dbc.CardBody(
            [
                html.H4("Parkinson's Identification", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",
                ),
                dbc.Button("Go somewhere", color="primary"),
            ]
        ),
    ],
    style={"width": "18rem", "display":"inline-block","margin":"1rem"},
)
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
        html.H1(
            [
                "Home Page"
            ]
        )
        ,
        html.Div([
            dbc.Row([
                        dbc.Col(html.Div(html.Img(src="assets/head_shot.jpg", height=250,
                            style={
                                # "width":"40%", 
                                "border-radius":"50%", 
                                "display":"block",
                                "margin-left":"auto",
                                "margin-right":"1em",
                                "margin-top":"1em",
                                "border":"4px solid white"}
                                )),
                            width=4),
                        dbc.Col(html.Div([
                            html.H1("Allan Khariton",style={"color":"white", "text-align":"right", "margin-top":"1em"}),
                            html.H1("Data Science Portfolio",style={"color":"white", "text-align":"right","align":"center"}),
                            html.Hr(style={"border":"3px solid white", "border-radius":"5px","color":"white"}),
                            html.H5("I'm a Data Analyst II, and advancing in my career as a data professional. My portfolio focuses on interesting projects I've recently undertaken, with a strong emphasis on business impact and learning new tools & languages. You can view my projects in the posts below, and visit my Github & LinkedIn pages (or download my Resume) by using the links below.",
                                        style={"color":"white", "font-weight":"normal"})
                        ]),width=6,style={})
                        
                    ],style={"background":"#101010"})
        ])
    ])
    return layout