import dash
from dash import html, dcc, register_page  #, callback # If you need callbacks, import it here.
import dash_bootstrap_components as dbc
from card import create_card_left, create_card_right



# register page with page name
register_page(
    __name__,
    name='Home',
    top_nav=True,
    path='/'
)


############
# content of the python tab, including all relevant cards for projects
#=############
python_tab = html.Div([
    create_card_left(
            image="https://github.com/akstl1/portfolio-deployments/blob/main/img/Park2.jpg?raw=true",
            title="Heart Disease Classification",
            description="Some quick example text to build on the card title and",
            url="/heart_disease_classification"
            ),
    create_card_right(
            image="https://github.com/akstl1/portfolio-deployments/blob/main/img/Park2.jpg?raw=true",
            title="Parkinson's Identification",
            description="Some quick example text to build on the card title and",
            url="/heart_disease_classification"),
    create_card_left(
            image="https://github.com/akstl1/portfolio-deployments/blob/main/img/Park2.jpg?raw=true",
            title="Telecom Customer Churn Prediction",
            description="Some quick example text to build on the card title and",
            url="/BI_aggregation"),
    create_card_right(
            image="https://github.com/akstl1/portfolio-deployments/blob/main/img/Park2.jpg?raw=true",
            title="Analyducks",
            description="Some quick example text to build on the card title and",
            url="/analyducks")
],style={"background-color":"#2C2F36"})

############
# content of the BI tab, including all relevant cards for projects
#=############
bi_tab = html.Div([
    create_card_left(
            image="https://github.com/akstl1/portfolio-deployments/blob/main/img/Park2.jpg?raw=true",
            title="Heart Disease Classification",
            description="Some quick example text to build on the card title and",
            url="/heart_disease_classification"
            )
],style={"background-color":"#2C2F36"})

############
# layout of the app, referencing the above tab content variables
#=############
def layout():
    layout = html.Div([
        html.Div([
                    # html.Hr(),
                    html.Div([
                        dbc.Tabs([
                                dbc.Tab(python_tab,label="Python",className="custom-tab",active_tab_class_name='custom-tab--selected', tab_style={"width":"33%"}),
                                dbc.Tab(bi_tab, label="Power BI", className="custom-tab",active_tab_class_name='custom-tab--selected',tab_style={"width":"33%"}),
                                dbc.Tab(bi_tab, className="custom-tab",active_tab_class_name='custom-tab--selected',label="Tableau", tab_style={"width":"33%"}),
                        ],style={"background-color":"#adadad","font-weight":"bold","height":"44px"})
                    ]),
        ], style={'height':'300px',"background-color":"#adadad"})
        
    ],style={"background-color":"#adadad"})
    return layout