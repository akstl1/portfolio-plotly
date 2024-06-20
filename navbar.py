from dash import html
import dash_bootstrap_components as dbc

############
# creates the navbar visible at the top of each page
##############

############
# navbar components
# 1 - left section with photo
# 2 - right section with name, title, description/career goals
# 3 - second navbar with links to home page, location, and professional media links
#=############
def create_navbar_new():
    navbar_new = html.Div([
                        html.Div([
                            html.Div([
                                html.Img(src="assets/head_shot.jpg", height=250, className="navImage",
                                            style={
                                                "border-radius":"50%", 
                                                # "display":"block",
                                                "margin-left":"auto",
                                                "margin-right":"1em",
                                                "margin-top":"1em",
                                                "border":"4px solid white",
                                                
                                            }
                                )],className="navImgCont",style={
                                    "display":"inline-block",
                                    # "width":"33%",
                                    # "background-color":"red"
                                    }),

                            html.Div([
                                html.H1("Allan Khariton",style={"color":"white", "text-align":"right", "margin-top":"1em", "padding-left":"1em", "padding-right":"1em"}),
                                html.H1("Data Science & Analytics Portfolio",style={"color":"white", "text-align":"right","align":"center", "padding-left":"1em", "padding-right":"1em"}),
                                html.Hr(style={"border":"3px solid white", "border-radius":"5px","color":"white", "margin-left":"1em", "margin-right":"1em"}),
                                html.H6("I'm a Data & BI Analyst II, and advancing in my career as a data professional. My portfolio focuses on interesting projects I've recently undertaken, with a strong emphasis on business impact and learning new tools & languages. You can view my projects in the posts below, and visit my Github & LinkedIn pages (or download my Resume) by using the links below.",
                                        style={"color":"white", "font-weight":"normal","display":"inline-block", "padding-left":"1em", "padding-right":"1em"}),


                                dbc.NavbarSimple(
                            children=[
                                dbc.NavItem(
                                    dbc.NavLink(
                                        [
                                            html.I(className="fa-solid fa-location-dot"),  # Font Awesome Icon
                                            " Rockville, MD"  # Text beside icon
                                        ],
                                    )
                                    # ,style={"margin-right":"3em"}

                                ),
                                dbc.NavItem(
                                    dbc.NavLink(
                                        [
                                            html.I(className="fa-brands fa-github"),  # Font Awesome Icon
                                            " Github"  # Text beside icon
                                        ],
                                        href="https://github.com/akstl1",
                                        target="_blank"
                                    )
                                    # ,style={"margin-right":"3em"}

                                ),
                                dbc.NavItem(
                                    dbc.NavLink(
                                        [
                                            html.I(className="fa-brands fa-linkedin"),  # Font Awesome Icon
                                            " LinkedIn"  # Text beside icon
                                        ],
                                        href="https://www.linkedin.com/in/allan-khariton/",
                                        target="_blank"
                                    )
                                    # ,style={"margin-right":"3em"}

                                ),
                                dbc.NavItem(
                                    dbc.NavLink(
                                        [
                                            html.I(className="fa-regular fa-file"),  # Font Awesome Icon
                                            " Resume"  # Text beside icon
                                        ],
                                        href="/assets/A_Khariton_Resume_2024.06.19.pdf",
                                        target="_blank",
                                        external_link=True,
                                    )
                                    # ,style={"margin-right":"3em"}

                                )
            
                            ],
                                brand='Home',
                                brand_href="/",
                                sticky="top",  # Uncomment if you want the navbar to always appear at the top on scroll.
                                color="#101010",  # Change this to change color of the navbar e.g. "primary", "secondary" etc.
                                dark=True,  # Change this to change color of text within the navbar (False for dark text)
                                style={"background":"#101010"}
                            )
                            ], className="navDesc", 
                            style={
                                # "display":"inline-block",
                                # "width":"50%"
                                # "background-color":"green"
                                })
                        ],className="navContainer",style={
                            # "display":"flex"
                            }),

                        
                        
                        



                        # dbc.Col(html.Div(html.Img(src="assets/head_shot.jpg", height=250, className="navImage",
                        #     style={
                        #         "border-radius":"50%", 
                        #         "display":"block",
                        #         "margin-left":"auto",
                        #         "margin-right":"1em",
                        #         "margin-top":"1em",
                        #         "border":"4px solid white"}
                        #         )),
                        #     width=4
                        #     ),
                        # dbc.Col(html.Div([
                        #     html.H1("Allan Khariton",style={"color":"white", "text-align":"right", "margin-top":"1em"}),
                        #     html.H1("Data Science & Analytics Portfolio",style={"color":"white", "text-align":"right","align":"center"}),
                        #     html.Hr(style={"border":"3px solid white", "border-radius":"5px","color":"white"}),
                        #     html.H6("I'm a Data & BI Analyst II, and advancing in my career as a data professional. My portfolio focuses on interesting projects I've recently undertaken, with a strong emphasis on business impact and learning new tools & languages. You can view my projects in the posts below, and visit my Github & LinkedIn pages (or download my Resume) by using the links below.",
                        #                 style={"color":"white", "font-weight":"normal"})
                        # ]),
                        # width=6,style={}
                        # ),
                        # dbc.Col(style={"padding":"0px","background":"#101010"}),




                    #     dbc.Row([


                    #     # second navbar with links
                    #     dbc.NavbarSimple(
                    #         children=[
                    #             dbc.NavItem(
                    #                 dbc.NavLink(
                    #                     [
                    #                         html.I(className="fa-solid fa-location-dot"),  # Font Awesome Icon
                    #                         " Rockville, MD"  # Text beside icon
                    #                     ],
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
                    #                     href="/assets/A_Khariton_Resume_2024.06.19.pdf",
                    #                     target="_blank",
                    #                     external_link=True
                    #                 ),style={"margin-right":"3em"}

                    #             )
            
                    #         ],
                    #             brand='Home',
                    #             brand_href="/",
                    #             sticky="top",  # Uncomment if you want the navbar to always appear at the top on scroll.
                    #             color="dark",  # Change this to change color of the navbar e.g. "primary", "secondary" etc.
                    #             dark=True,  # Change this to change color of text within the navbar (False for dark text)
                    #         )
                        
                    # ])
    ],style={"background":"#101010"})

    return navbar_new