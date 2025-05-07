from dash import Dash
from .layout import serve_layout
from .callbacks import register_callbacks

def create_dash_app():
    app = Dash(__name__)

    # serve layout and register callback functions
    app.layout = serve_layout()
    register_callbacks(app)

    return app
