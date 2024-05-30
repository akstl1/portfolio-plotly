from dash import html
import dash_bootstrap_components as dbc

def create_card(image,title,description,position,url):
    if position=="left":
        left="10%"
        right="5%"
    else:
        left="5%"
        right="10%"
    card = dbc.Card(
    [
        dbc.CardImg(src=image, top=True),
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
    style={"width": "auto","display":"inline-block", "margin-top":"2em", "margin-bottom":"1em","margin-left":left,"margin-right":right},
)
# def create_navbar():
    # navbar = dbc.NavbarSimple(
    #     children=[
    #         dbc.NavItem(
    #             dbc.NavLink(
    #                 [
    #                     html.I(className="fa-brands fa-github"),  # Font Awesome Icon
    #                     " "  # Text beside icon
    #                 ],
    #                 href="[YOUR GITHUB PROFILE URL]",
    #                 target="_blank"
    #             )

    #         ),
    #         dbc.NavItem(
    #             dbc.NavLink(
    #                 [
    #                     html.I(className="fa-brands fa-medium"),  # Font Awesome Icon
    #                     " "  # Text beside icon
    #                 ],
    #                 href="[YOUR MEDIUM PROFILE URL]",
    #                 target="_blank"
    #             )

    #         ),
    #         dbc.NavItem(
    #             dbc.NavLink(
    #                 [
    #                     html.I(className="fa-brands fa-linkedin"),  # Font Awesome Icon
    #                     " "  # Text beside icon
    #                 ],
    #                 href="[YOUR LINKEDIN PROFILE URL]",
    #                 target="_blank"
    #             )

    #         ),
    #         dbc.DropdownMenu(
    #             nav=True,
    #             in_navbar=True,
    #             label="Menu",
    #             align_end=True,
    #             children=[  # Add as many menu items as you need
    #                 dbc.DropdownMenuItem("Home", href='/'),
    #                 dbc.DropdownMenuItem(divider=True),
    #                 dbc.DropdownMenuItem("Page 2", href='/page2'),
    #                 dbc.DropdownMenuItem("Page 3", href='/page3'),
    #             ],
    #         ),
    #     ],
    #     brand='Home',
    #     brand_href="/",
    #     # sticky="top",  # Uncomment if you want the navbar to always appear at the top on scroll.
    #     color="dark",  # Change this to change color of the navbar e.g. "primary", "secondary" etc.
    #     dark=True,  # Change this to change color of text within the navbar (False for dark text)
    # )


# def create_navbar_new():
#     navbar_new = html.Div([dbc.Row([
#                         dbc.Col(html.Div(html.Img(src="assets/head_shot.jpg", height=250,
#                             style={
#                                 # "width":"40%", 
#                                 "border-radius":"50%", 
#                                 "display":"block",
#                                 "margin-left":"auto",
#                                 "margin-right":"1em",
#                                 "margin-top":"1em",
#                                 "border":"4px solid white"}
#                                 )),
#                             width=4),
#                         dbc.Col(html.Div([
#                             html.H1("Allan Khariton",style={"color":"white", "text-align":"right", "margin-top":"1em"}),
#                             html.H1("Data Science Portfolio",style={"color":"white", "text-align":"right","align":"center"}),
#                             html.Hr(style={"border":"3px solid white", "border-radius":"5px","color":"white"}),
#                             html.H5("I'm a Data Analyst II, and advancing in my career as a data professional. My portfolio focuses on interesting projects I've recently undertaken, with a strong emphasis on business impact and learning new tools & languages. You can view my projects in the posts below, and visit my Github & LinkedIn pages (or download my Resume) by using the links below.",
#                                         style={"color":"white", "font-weight":"normal"})
#                         ]),width=6,style={}),
#                         dbc.Col(style={"padding":"0px","background":"#101010"}),
#                         dbc.NavbarSimple(
#         children=[
#             dbc.NavItem(
#                 dbc.NavLink(
#                     [
#                         html.I(className="fa-solid fa-location-dot"),  # Font Awesome Icon
#                         " Rockville, MD"  # Text beside icon
#                     ],
#                     href="",
#                     target="_blank"
#                 ),style={"margin-right":"3em"}

#             ),
#             dbc.NavItem(
#                 dbc.NavLink(
#                     [
#                         html.I(className="fa-brands fa-github"),  # Font Awesome Icon
#                         " Github"  # Text beside icon
#                     ],
#                     href="https://github.com/akstl1",
#                     target="_blank"
#                 ),style={"margin-right":"3em"}

#             ),
#             dbc.NavItem(
#                 dbc.NavLink(
#                     [
#                         html.I(className="fa-brands fa-linkedin"),  # Font Awesome Icon
#                         " LinkedIn"  # Text beside icon
#                     ],
#                     href="https://www.linkedin.com/in/allan-khariton/",
#                     target="_blank"
#                 ),style={"margin-right":"3em"}

#             ),
#             dbc.NavItem(
#                 dbc.NavLink(
#                     [
#                         html.I(className="fa-regular fa-file"),  # Font Awesome Icon
#                         " Resume"  # Text beside icon
#                     ],
#                     href="[YOUR LINKEDIN PROFILE URL]",
#                     target="_blank"
#                 ),style={"margin-right":"3em"}

#             )
            
#         ],
#         brand='Home',
#         brand_href="/",
#         sticky="top",  # Uncomment if you want the navbar to always appear at the top on scroll.
#         color="dark",  # Change this to change color of the navbar e.g. "primary", "secondary" etc.
#         dark=True,  # Change this to change color of text within the navbar (False for dark text)
#     )
                        
#                     ])
#     ],style={"background":"#101010"})

    return card