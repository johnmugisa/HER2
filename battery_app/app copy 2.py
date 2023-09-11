import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import os
import plotly.io as pio

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    html.Div(dcc.Graph(id='plot-display'), style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '80vh'})
], style={'textAlign': 'center'})

dcc.Dropdown(
        id='plot-type-dropdown',
        options=[
            {'label': 'Capacity vs. Cycle', 'value': 'capacity-cycle'},
            {'label': 'Voltage vs. Cycle', 'value': 'voltage-cycle'}
        ],
        value='capacity-cycle'
    ),
    
dcc.Graph(id='plot-display', style={"height": "100vh", "width": "100vw"}),
html.Button('Save Plot', id='save-plot-button'),
html.Div(id='save-output', children=[])


@app.callback(
    Output('output-data-upload', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified')]
)
def update_output(content, name, date):
    if content is not None:
        children = [
            html.Div(f'User uploaded {name} on {date}')
        ]
        return children



@app.callback(
    Output('plot-display', 'figure'),
    [Input('plot-type-dropdown', 'value'),
     Input('upload-data', 'contents')]
)
def display_selected_plot(plot_type, contents):
    dummy_data = {"cycle": list(range(10)), "capacity": list(range(10)), "voltage": [i*2 for i in range(10)]}
    df = pd.DataFrame(dummy_data)

    if plot_type == "capacity-cycle":
        fig = px.line(df, x='cycle', y='capacity')
    elif plot_type == "voltage-cycle":
        fig = px.line(df, x='cycle', y='voltage')
    else:
        fig = px.line(df)
    
    # Customizing the figure
    fig.update_layout(
        font=dict(
            family="Arial",
            size=12,
            color="black"
        ),
        xaxis=dict(
            linewidth=1.2,
            linecolor='black',
            mirror=True
        ),
        yaxis=dict(
            linewidth=1.2,
            linecolor='black',
            mirror=True
        )
    )
    return fig

@app.callback(
    Output('save-output', 'children'),
    [Input('save-plot-button', 'n_clicks')],
    [State('plot-display', 'figure')]
)
def save_plot(n_clicks, current_figure):
    if n_clicks is not None:
        # Print the current working directory
        print("Current working directory:", os.getcwd())
        
        # Ensure 'saved_plots' directory exists
        if not os.path.exists('saved_plots'):
            print("saved_plots directory does not exist. Creating it now.")
            os.mkdir('saved_plots')
        else:
            print("saved_plots directory already exists.")
            
        # Specify where to save the plot
        path = os.path.join("saved_plots", f"saved_plot_{n_clicks}.png")
        
        # Save the plot
        pio.write_image(current_figure, path)
        
        return html.Div(f'Plot saved as {path}')




if __name__ == '__main__':
    app.run_server(debug=True)
