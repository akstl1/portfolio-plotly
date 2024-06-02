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
from card import create_card_left,create_card_right
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
cards = "/assets/cardStyle.css"
tabs = 'https://codepen.io/chriddyp/pen/bWLwgP.css'
app = dash.Dash(
    __name__,
    # suppress_callback_exceptions=True,

    external_stylesheets=[
        dbc.themes.BOOTSTRAP,  # Dash Themes CSS
        # dbc.themes.BOOTSTRAP,
        FA621,  # Font Awesome Icons CSS
        cards
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

app.layout = html.Div(
            [
                NAVBAR,
                dash.page_container
            ]
        )

#######
## Old layout, which reloaded whole page on callbacks
#######

# dcc.Loading(  # <- Wrap App with Loading Component
#     id='loading_page_content',
#     children=[
#         html.Div(
#             [
#                 NAVBAR,
#                 dash.page_container
#             ]
#         )
#     ],
#     color='primary',  # <- Color of the loading spinner
#     fullscreen=True  # <- Loading Spinner should take up full screen
# )

#######
## Old layout, which reloaded whole page on callbacks
#######

server = app.server


if __name__ == '__main__':
    app.run_server(debug=False)