from dash import Dash, html

app = Dash(__name__)
app.layout = html.Div(children=['Hello, world!', 'This is a test for CD'])

if __name__ == '__main__':
    app.run(debug=False)
