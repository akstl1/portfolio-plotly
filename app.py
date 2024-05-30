import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from navbar import create_navbar_new
from dash import html
from dash import dcc
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import requests
import plotly.express as px
from card import create_card
# Toggle the themes at [dbc.themes.LUX]
# The full list of available themes is:
# BOOTSTRAP, CERULEAN, COSMO, CYBORG, DARKLY, FLATLY, JOURNAL, LITERA, LUMEN,
# LUX, MATERIA, MINTY, PULSE, SANDSTONE, SIMPLEX, SKETCHY, SLATE, SOLAR,
# SPACELAB, SUPERHERO, UNITED, YETI, ZEPHYR.
# To see all themes in action visit:
# https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/explorer/

NAVBAR = create_navbar_new()
# CARD = create_card()
# To use Font Awesome Icons
FA621 = "https://use.fontawesome.com/releases/v6.2.1/css/all.css"
APP_TITLE = "Allan K Portfolio"

app = dash.Dash(
    __name__,
    # suppress_callback_exceptions=True,

    external_stylesheets=[
        dbc.themes.BOOTSTRAP,  # Dash Themes CSS
        # dbc.themes.BOOTSTRAP,
        FA621,  # Font Awesome Icons CSS
    ],
    title=APP_TITLE,
    use_pages=True,  
    # New in Dash 2.7 - Allows us to register pages
)

# # To use if you're planning on using Google Analytics
# app.index_string = f'''
# <!DOCTYPE html>
# <html>
#     <head>
#         {{%metas%}}
#         <title>{APP_TITLE}</title>
#         {{%favicon%}}
#         {{%css%}}
#     </head>
#     <body>
#         {{%app_entry%}}
#         <footer>
#             {{%config%}}
#             {{%scripts%}}
#             {{%renderer%}}
#         </footer>
        
#     </body>
# </html>
# '''

app.layout = dcc.Loading(  # <- Wrap App with Loading Component
    id='loading_page_content',
    children=[
        html.Div(
            [
                NAVBAR,
                dash.page_container
            ]
        )
    ],
    color='primary',  # <- Color of the loading spinner
    fullscreen=True  # <- Loading Spinner should take up full screen
)

server = app.server

@app.callback(Output('pokemon-name-id','children'),
              Output('pokemon-description','children'),
              Output('pokemon-ability','children'),
              Output('pokemon-type','children'),
              Output('pokemon-height','children'),
              Output('pokemon-weight','children'),
              Output('pokemon-sprite','src'),
              Output('pokemon-sprite','style'),
                [Input('pokemon-name', 'value')])

def name_and_id(poke_input):

    ## Pokemon Species Data Request
    pokemon_species_request = requests.get("https://pokeapi.co/api/v2/pokemon-species/"+str(poke_input)+"/")
    species_data = pokemon_species_request.json()

    ## Pokemon  Table Data Request
    pokemon_request = requests.get("https://pokeapi.co/api/v2/pokemon/"+str(poke_input)+"/")
    pokemon_data = pokemon_request.json()

    ## Name And Id callback_
    name=species_data['name'].capitalize()
    id=str(species_data['id'])
    while len(id)<3:
        id='0'+id

    ## Description callback
    entry=species_data['flavor_text_entries'][0]['flavor_text'].replace('\x0c',' ')

    ## Ability Data
    abilities_json=pokemon_data['abilities']
    abilities = []
    for ability in abilities_json:
        abilities.append(ability['ability']['name'].capitalize())

    ## Types Data
    types_json=pokemon_data['types']
    types = []
    for type in types_json:
        types.append(type['type']['name'].capitalize())

    ## Height Data
    height=pokemon_data['height']/10

    ## Weight Data
    weight=pokemon_data['height']/10

    ## Sprite Data_
    id=str(species_data['id'])

    ## return statement
    return "{} #{}".format(name, id),"Description: {}".format(entry),"Abilities: "+', '.join(abilities),"Types: "+', '.join(types),"Height: {} m".format(height),"Weight: {} kg".format(weight),"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/"+id+".png", {'width':'275px', 'text-align':'center'}

# create callback and function to generate base stats graph

@app.callback(Output('graph', 'figure'),
              Output('graph', 'style'),
              [Input('pokemon-name','value')])

def update_figure(poke_input):
    #get data
    poke_request = requests.get("https://pokeapi.co/api/v2/pokemon/"+str(poke_input)+"/")
    json_data = poke_request.json()
    stats_json=json_data['stats']
    stats=[]
    #cycle through data and append it to the stats list
    for stat in stats_json:
        stats.append([stat['stat']['name'], stat['base_stat']])
    # generate df with the stats list data, generate bar plot
    df = pd.DataFrame(stats, columns = ['Stat', 'Base Value'])
    fig = px.bar(df, x="Stat", y="Base Value",text_auto=True)
    fig.update_yaxes(range=[0, 270])
    return fig,{'height':'300px'}

if __name__ == '__main__':
    app.run_server(debug=False)