
import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import requests
import plotly.express as px

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server=app.server


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



# create app layout
# dummy line of text
app.layout = html.Div([
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
                # html.Div([dcc.Dropdown(id='pokemon-name',options=[{'label':i.capitalize(),'value':i} for i in poke_names_list], value='bulbasaur')],style={'width':'20%', 'margin-left':'auto','margin-right':'auto'}),
                # html.Div([html.H1(id='pokemon-name-id')], style={'text-align':'center'}),
                # html.Div([
                #     html.Div([html.Img(id="pokemon-sprite")],style={'display':'inline-block', 'width':'20%','height':'300px', 'margin-right':'60px','margin-left':'80px', 'text-align':'center','vertical-align':'top' }),
                #     html.Div([
                #         html.Div([html.P(id='pokemon-description'),
                #         html.Div([
                #             html.Div([html.P(id='pokemon-height')]),
                #             html.Div([html.P(id='pokemon-weight')])
                #             ])
                #             ]),
                #             html.P(id='pokemon-ability'),
                #             html.P(id='pokemon-type')], style={'display':'inline-block', 'width':'30%','height':'300px','background-color':'#30a7d7', 'vertical-align':'top', 'padding-left':'10px','padding-right':'10px', 'border-radius':'10px'}),
                #             html.Div([dcc.Graph(id='graph')], style={'display':'inline-block','width':'30%', 'margin-left':'40px'})

                    ], style={'height':'300px'})


])
# , style={'background-color':'LightCyan', 'padding-bottom':'275px'})

#create callback to get pokemon stats for above elements

# @app.callback(Output('pokemon-name-id','children'),
#               Output('pokemon-description','children'),
#               Output('pokemon-ability','children'),
#               Output('pokemon-type','children'),
#               Output('pokemon-height','children'),
#               Output('pokemon-weight','children'),
#               Output('pokemon-sprite','src'),
#               Output('pokemon-sprite','style'),
#                 [Input('pokemon-name', 'value')])


# def name_and_id(poke_input):

#     ## Pokemon Species Data Request
#     pokemon_species_request = requests.get("https://pokeapi.co/api/v2/pokemon-species/"+str(poke_input)+"/")
#     species_data = pokemon_species_request.json()

#     ## Pokemon  Table Data Request
#     pokemon_request = requests.get("https://pokeapi.co/api/v2/pokemon/"+str(poke_input)+"/")
#     pokemon_data = pokemon_request.json()

#     ## Name And Id callback_
#     name=species_data['name'].capitalize()
#     id=str(species_data['id'])
#     while len(id)<3:
#         id='0'+id

#     ## Description callback
#     entry=species_data['flavor_text_entries'][0]['flavor_text'].replace('\x0c',' ')

#     ## Ability Data
#     abilities_json=pokemon_data['abilities']
#     abilities = []
#     for ability in abilities_json:
#         abilities.append(ability['ability']['name'].capitalize())

#     ## Types Data
#     types_json=pokemon_data['types']
#     types = []
#     for type in types_json:
#         types.append(type['type']['name'].capitalize())

#     ## Height Data
#     height=pokemon_data['height']/10

#     ## Weight Data
#     weight=pokemon_data['height']/10

#     ## Sprite Data_
#     id=str(species_data['id'])

#     ## return statement
#     return "{} #{}".format(name, id),"Description: {}".format(entry),"Abilities: "+', '.join(abilities),"Types: "+', '.join(types),"Height: {} m".format(height),"Weight: {} kg".format(weight),"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/"+id+".png", {'width':'275px', 'text-align':'center'}

# # create callback and function to generate base stats graph

# @app.callback(Output('graph', 'figure'),
#               Output('graph', 'style'),
#               [Input('pokemon-name','value')])
# def update_figure(poke_input):
#     #get data
#     poke_request = requests.get("https://pokeapi.co/api/v2/pokemon/"+str(poke_input)+"/")
#     json_data = poke_request.json()
#     stats_json=json_data['stats']
#     stats=[]
#     #cycle through data and append it to the stats list
#     for stat in stats_json:
#         stats.append([stat['stat']['name'], stat['base_stat']])
#     # generate df with the stats list data, generate bar plot
#     df = pd.DataFrame(stats, columns = ['Stat', 'Base Value'])
#     fig = px.bar(df, x="Stat", y="Base Value",text_auto=True)
#     fig.update_yaxes(range=[0, 270])
#     return fig,{'height':'300px'}

if __name__=="__main__":
    app.run_server()
