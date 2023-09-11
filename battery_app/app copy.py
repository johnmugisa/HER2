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
import matplotlib.pyplot as plt

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
    dcc.Dropdown(
        id='plot-type-dropdown',
        options=[
            {'label': 'Capacity vs. Cycle', 'value': 'capacity-cycle'},
            {'label': 'Voltage vs. Cycle', 'value': 'voltage-cycle'},
            {'label': 'Voltage vs. State of Charge (SoC)', 'value': 'voltage-State of Charge (SoC)'}
        ],
        value='capacity-cycle'
    ),
    html.Div(dcc.Graph(id='plot-display'), style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '80vh'}),
    html.Button('Save Plot', id='save-plot-button'),
    html.Div(id='save-output')
], style={'textAlign': 'center'})

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
            # Replace this section with your actual column names and data processing code
            columns = ["cycle number", "Q discharge/mA.h", "Q charge/mA.h", "Ewe/V", "(-Qo)/mA.h", "time/s", "<I>/mA","Capacity/mA.h","Efficiency/%"]
            data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t', skiprows=1, header=None, names=columns, engine='python')
            data = data.apply(lambda col: pd.to_numeric(col, errors='coerce'))
            # After reading in the data
            print(data)
       


            unique_cycles = data['cycle number'].unique()
            # Remove NaN values and get first 8 cycles
            cycle_number= unique_cycles[~np.isnan(unique_cycles)][:8]
            colors = plt.cm.viridis(np.linspace(0, 1, len(cycle_number)))
            color_strings = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors]
            
            fig = go.Figure()

            for index, cycle_number in enumerate(cycle_number):
                cycle_data = data[data['cycle number'] == cycle_number]
    
                if cycle_data.shape[0] == 0:
                    print(f"No data available for cycle number {cycle_number}.")
                    continue

                charge_values = cycle_data['Q charge/mA.h'].values[:-945]
                discharge_values = cycle_data['Q discharge/mA.h'].values[500:-5]
    
                if len(charge_values) == 0 or len(discharge_values) == 0:
                    print(f"Insufficient data for cycle number {cycle_number}.")
                    continue

                max_charge = np.max(charge_values)
                SoC_charge = charge_values / max_charge

                max_discharge = np.max(discharge_values)
                SoC_discharge = discharge_values / max_discharge

                Ewe_V_charge = cycle_data['Ewe/V'].to_numpy()[:-945]
                Ewe_V_discharge = cycle_data['Ewe/V'].to_numpy()[500:-5]


    # ... (continue the rest of your plotting code)

            # Customizing the figure

            # Customizing the figure
            fig.update_layout(
                title="Potential vs. SoC for Multiple Cycles",
                xaxis_title="State of Charge (SoC)",
                yaxis_title="Ewe/V",
                height=600,
                width=800,
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
        # Ensure 'saved_plots' directory exists
        if not os.path.exists('saved_plots'):
            os.mkdir('saved_plots')
        
        # Specify where to save the plot
        path = os.path.join("saved_plots", f"saved_plot_{n_clicks}.png")
        
        # Save the plot
        pio.write_image(current_figure, path)
        
        return html.Div(f'Plot saved as {path}')

if __name__ == '__main__':
    app.run_server(debug=True)
