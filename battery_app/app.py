import dash
from dash import dcc, html, Output, Input, State
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
    html.Div([
        html.Div(id='output-data-upload'),
        html.Div(
            dcc.Graph(id='plot-display', style={'height': '80vh'}),
            style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'height': '80vh'
        }
    ),
    html.Button('Save Plot', id='save-plot-button'),
    html.Div(id='save-output')
], style={'textAlign': 'center'}) ])

@app.callback(
    Output('output-data-upload', 'children'),
    Output('plot-display', 'figure'),
    [Input('upload-data', 'contents')],
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return None, {}

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'txt' in filename:
            # Define the column names
            columns = ["cycle number", "Q discharge/mA.h", "Q charge/mA.h", "Ewe/V", "(-Qo)/mA.h", "time/s", "<I>/mA", "Capacity/mA.h", "Efficiency/%"]

            # Read the data, specifying the delimiter as a tab
            data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t', skiprows=1, header=None, names=columns, engine='python')

            # Convert the comma to a dot
            # data = data.applymap(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
            # Convert the comma to a dot if the numbers are in the European format
            data = data.apply(lambda x: x.str.replace(',', '.') if x.dtype == "object" else x).astype(float)

            # List of cycle numbers to plot
            cycles_to_plot = [2,3,4,5,6,7,8,9]

            # Define a colormap to get different colors for each cycle
            colors = [
                "blue",
                "green",
                "red",
                "cyan",
                "magenta",
                "yellow",
                "black",
                "purple",
                "pink",
            ]

            fig = go.Figure()

            for index, cycle_number in enumerate(cycles_to_plot):
                cycle_data = data[data['cycle number'] == cycle_number]

                if cycle_data.empty:
                    print(f"No data available for cycle number {cycle_number}.")
                    continue
                
                max_charge = np.max(cycle_data['Q charge/mA.h'].values[:-945])
                SoC_charge = cycle_data['Q charge/mA.h'].values[:-945] / max_charge

                max_discharge = np.max(cycle_data['Q discharge/mA.h'].values[500:-5])
                SoC_discharge = cycle_data['Q discharge/mA.h'].values[500:-5] / max_discharge

                Ewe_V_charge = cycle_data['Ewe/V'].to_numpy()[:-945]
                Ewe_V_discharge = cycle_data['Ewe/V'].to_numpy()[500:-5]

                fig.add_trace(go.Scatter(x=SoC_charge, y=Ewe_V_charge, mode='lines', line=dict(color=colors[index], width=2, dash='dash'), name=f'Cycle {cycle_number} Charge'))
                fig.add_trace(go.Scatter(x=SoC_discharge, y=Ewe_V_discharge, mode='lines', line=dict(color=colors[index], width=2), name=f'Cycle {cycle_number} Discharge'))

                        # Customizing the figure
            fig.update_layout(
                title="Potential vs. SoC for Multiple Cycles",
                xaxis_title="State of Charge (SoC)",
                yaxis_title="Ewe/V",
                height=400,
                width=600,
                font=dict(
                    family="Arial",
                    size=13,
                    color="black"
                ),
                xaxis=dict(
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    mirror=True,
                    showgrid=False,
                    tickfont=dict(
                        family="serif",
                        size=14
                    ),
                    title_font=dict(
                        size=16
                    ),
                    ticks="outside",
                    tickwidth=2,
                    tickcolor="black",
                    ticklen=5,
                    title_standoff=15
                ),
                yaxis=dict(
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    mirror=True,
                    showgrid=False,
                    tickfont=dict(
                        family="serif",
                        size=14
                    ),
                    title_font=dict(
                        size=16
                    ),
                    ticks="outside",
                    tickwidth=2,
                    tickcolor="black",
                    ticklen=5,
                    title_standoff=15
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=True
            )

            return f'File "{filename}" processed and plotted.', fig


            return f'File "{filename}" processed and plotted.', fig

        else:
            return 'Unsupported File Type', {}

    except Exception as e:
        print(e)
        return f'Error processing "{filename}".', {}

@app.callback(
    Output('save-output', 'children'),
    [Input('save-plot-button', 'n_clicks')],
    [State('plot-display', 'figure')]
)
def save_plot(n_clicks, current_figure):
    if n_clicks is not None:
        if not os.path.exists('saved_plots'):
            os.mkdir('saved_plots')
        
        path = os.path.join("saved_plots", f"saved_plot_{n_clicks}.png")
        
        pio.write_image(current_figure, path)
        
        return html.Div(f'Plot saved as {path}')

if __name__ == '__main__':
    app.run_server(debug=True)
