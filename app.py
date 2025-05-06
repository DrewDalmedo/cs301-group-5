from dash import Dash, html, dcc, Input, Output, State, callback_context
import pandas as pd
import io
import base64

app = Dash(__name__)
#app.layout = html.Div(children=['Hello, world!'])
app.layout = html.Div([
    html.H3("Upload Your Dataset"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center',
            'margin': '10px'
        },
        
        multiple = False
    ),
    html.H3("Select Target Variable"),
    dcc.Dropdown(id = 'target-dropdown', placeholder='Select a target variable'),
    html.Div(id='selected-target-output'),
    dcc.Store(id='selected-target'),


    html.Div(id='upload-status'),
    dcc.Store(id='stored-data'),

    html.H3("Select Categorical Variable"),
    dcc.RadioItems(id='categorical-radio'),

    html.H3("Average Target Value per Category"),
    dcc.Graph(id = 'category-bar'),
    html.H3("Correlation with Target Variable"),
    dcc.Graph(id='correlation-bar')
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
if __name__ == '__main__':
    app.run(debug=False)
