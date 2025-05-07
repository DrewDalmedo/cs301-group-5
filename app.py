from dash import Dash, html, dcc, Input, Output, State, callback_context
import pandas as pd
import io
import base64
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from dash.dependencies import ALL
from sklearn.tree import DecisionTreeRegressor


app = Dash(__name__)
app.layout = html.Div([
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

@app.callback(
    Output('upload-status', 'children'),
    Output('stored-data', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    if contents is None:
        return '', None

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return f"Successfully uploaded: {filename}", df.to_dict('records')
        else:
            return "Only CSV files are supported.", None
    except Exception as e:
        return f"Error processing file: {str(e)}", None

@app.callback(
        Output('target-dropdown', 'options'),
        Input('stored-data', 'data')
)

def update_target_dropdown(data):
    if data is None:
        return []
    df = pd.DataFrame(data)
    num_cols = df.select_dtypes(include='number').columns
    return [{'label': col, 'value': col} for col in num_cols]

@app.callback(
        Output('selected-target-output', 'children'),
        Output('selected-target', 'data'),
        Input('target-dropdown', 'value')

)
def store_selected_target(target):
    if target is None:
        return "", None
    return f"Selected target variable: {target}", target

@app.callback(
    Output('categorical-radio', 'options'),
    Input('stored-data', 'data')
)
def update_categorical_options(data):
    if data is None:
        return []
    df = pd.DataFrame(data)
    cat_cols = df.select_dtypes(include='object').columns
    return [{'label':col, 'value': col} for col in cat_cols]

@app.callback(
    Output('category-bar', 'figure'),
    Input('stored-data', 'data'),
    Input('categorical-radio', 'value'),
    Input('selected-target', 'data')
)
def update_category_bar(data, cat_col, target):
    if data is None or cat_col is None or target is None:
        return {}
    df = pd.DataFrame(data)
    avg_df = df.groupby(cat_col)[target].mean().reset_index()
    return {
        'data': [{
            'x': avg_df[cat_col],
            'y': avg_df[target],
            'type':'bar'
    }],
        'layout': {'title': f'Average {target} by {cat_col}'}
    }

@app.callback(
    Output('correlation-bar', 'figure'),
    Input('stored-data', 'data'),
    Input('selected-target', 'data')
)

def update_correlation_bar(data, target):
    if data is None or target is None:
        return {}
    df = pd.DataFrame(data)
    numeric_df = df.select_dtypes(include='number')
    if target not in numeric_df.columns:
        return {}
    corr = numeric_df.corr()[target].drop(target).abs().sort_values(ascending=False)
    return {
        'data': [{
            'x': corr.index,
            'y': corr.values,
            'type': 'bar'
        }],
        'layout': {'title': f'Correlation of Variables with {target}'}
    }
@app.callback(
    Output('feature-checkboxes', 'children'),
    Input('stored-data', 'data'),
    Input('selected-target', 'data')
)
def display_feature_boxes(data, target):
    if data is None or target is None:
        return []
    df = pd.DataFrame(data)
    feature_options = [col for col in df.columns if col != target]
    return dcc.Checklist(
        id='feature-list',
        options=[{'label': col, 'value': col} for col in feature_options],
        labelStyle={'display': 'block'}
    )
@app.callback(
    Output('model-output', 'children'),
    Output('trained-model', 'data'),
    Input('train-button', 'n_clicks'),
    State('stored-data', 'data'),
    State('selected-target', 'data'),
    State('feature-list', 'value')
)
def train_model(n_clicks, data, target, features):
    if n_clicks == 0 or data is None or target is None or features is None:
        return "", None
    
    df = pd.DataFrame(data)
    df = df.dropna(subset=[target])  #drop some rows to clean up a bit

    X= df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

    numeric_features = X.select_dtypes(include='number').columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy = 'mean'), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor), 
        ('regressor', DecisionTreeRegressor(random_state=0))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return f"Model trained! R² score: {r2:.4f}", None

@app.callback(
    Output('input-fields', 'children'),
    Input('feature-list', 'value'),
    State('stored-data', 'data')
)

def generate_input_fields(features, data):
    if features is None or data is None:
        return []
    
    df = pd.DataFrame(data)
    inputs = []

    for feature in features:
        if df[feature].dtype == 'object':
            unique_vals = df[feature].dropna().unique().tolist()
            inputs.append(html.Div([
                html.Label(f"{feature}:"),
                dcc.Dropdown(id={'type': 'predict-input', 'index': feature},
                             options=[{'label': val, 'value': val} for val in unique_vals],
                             placeholder=f"Select a value")
            ]))
        else:
            inputs.append(html.Div([
                html.Label(f"{feature}:"),
                dcc.Input(id={'type': 'predict-input', 'index': feature}, type='number')
            ]))
    return inputs

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('feature-list', 'value'),
    State('stored-data', 'data'),
    State('selected-target', 'data'),
    State({'type': 'predict-input', 'index': ALL}, 'value')
)
def make_prediction(n_clicks, features, data, target, inputs):
    if n_clicks == 0 or not features or not data or not target or not inputs:
        return ""

    df = pd.DataFrame(data)
    input_dict = dict(zip(features, inputs))
    input_df = pd.DataFrame([input_dict])

    numeric_features = input_df.select_dtypes(include='number').columns.tolist()
    categorical_features = input_df.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(random_state=0))
    ])

    X = df[features]
    y = df[target]
    pipeline.fit(X, y)

    prediction = pipeline.predict(input_df)[0]
    return f"Predicted {target}: {prediction:.2f}"

if __name__ == '__main__':
    app.run(debug=False)
