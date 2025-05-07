from dash import html, dcc

def serve_layout():
    return html.Div([
        html.H3("Upload Your Dataset"),
        dcc.Upload(
            id='upload-data',
            children=html.Div('Upload File'),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center',
                'margin': '10px'
            },
            
            multiple = False
        ),
        html.H3("Select Target: "),
        dcc.Dropdown(id = 'target-dropdown', placeholder='Select a target variable'),
        html.Div(id='selected-target-output'),
        dcc.Store(id='selected-target'),

        
        html.Div(id='upload-status'),
        dcc.Store(id='stored-data'),
        dcc.Store(id='trained-model'),


        html.H3("Select Categorical Variable"),
        dcc.RadioItems(id='categorical-radio'),

        html.Div([
            html.Div([
                html.H3("Average Target Value per Category"),
                dcc.Graph(id='category-bar')
            ], style={'width': '50%', 'padding': '0 10px'}),

            html.Div([
                html.H3("Correlation with Target Variable"),
                dcc.Graph(id='correlation-bar')
            ], style={'width': '50%', 'padding': '0 10px'})
        ], style={'display': 'flex', 'justify-content': 'space-between'}),


      

        html.H3("Train Model"),
        html.Div(id='feature-checkboxes'),
        html.Button("Train Model", id='train-button', n_clicks=0),
        html.Div(id='model-output'),

        html.H3("Make a Prediction"),
        html.Div(id='input-fields'),
        html.Button("Predict", id='predict-button', n_clicks=0),
        html.Div(id='prediction-output')
    ])

