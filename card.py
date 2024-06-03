from dash import html
import dash_bootstrap_components as dbc

### create two card classes, one for left and one for right side. these are identical except for margin styling that is set in css file via classL/R classNames

### cards will be used on the home page to display the project summaries for users to click on
### cards contain an image, title, description, and button with link to the project page

def create_card_left(image,title,description,url):
    card_left = dbc.Card(
    [
        dbc.CardImg(src=image, top=True,style={"border-top":"#2C2F36","border-top-left-radius":"2%","border-top-right-radius":"2%"}),
        dbc.CardBody(
            [
                html.H4(title, className="card-title"),
                html.P(
                    description,
                    className="card-text",
                ),
                dbc.Button("View Project", color="primary",href=url),
            ]
        ),
    ],
    style={"margin-top":"2em", "margin-bottom":"1em"
        #    ,"margin-left":left,"margin-right":right
           },
    className="cardL"
)
    return card_left

def create_card_right(image,title,description,url):
    card_right = dbc.Card(
    [
        dbc.CardImg(src=image, top=True,style={"border-top-left-radius":"2%","border-top-right-radius":"2%"}),
        dbc.CardBody(
            [
                html.H4(title, className="card-title"),
                html.P(
                    description,
                    className="card-text",
                ),
                dbc.Button("View Project", color="primary",href=url),
            ]
        ),
    ],
    style={"margin-top":"2em", "margin-bottom":"1em"
        #    ,"margin-left":left,"margin-right":right
           },
    className="cardR"
)
    return card_right