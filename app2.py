import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Sample data for skill levels before and after training
skills = ['SQL Injection', 'XSS', 'CSRF', 'Authentication', 'Authorization', 'Encryption', 'Logging', 'Monitoring']
before_training = [3, 2, 4, 3, 2, 3, 2, 3]
after_training = [5, 4, 5, 4, 4, 5, 4, 5]

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Cyclopt Training Progress Dashboard"),
    dcc.Graph(id='radar-chart'),
    html.Div(id='indicator-container', children=[
        html.Div([
            dcc.Graph(id='indicator-0'),
            dcc.Slider(
                id='slider-0',
                min=0,
                max=5,
                step=0.5,
                value=after_training[0],
                marks={i: str(i) for i in range(6)}
            ),
            html.Div(id='slider-output-container-0')
        ]),
        html.Div([
            dcc.Graph(id='indicator-1'),
            dcc.Slider(
                id='slider-1',
                min=0,
                max=5,
                step=0.5,
                value=after_training[1],
                marks={i: str(i) for i in range(6)}
            ),
            html.Div(id='slider-output-container-1')
        ]),
        html.Div([
            dcc.Graph(id='indicator-2'),
            dcc.Slider(
                id='slider-2',
                min=0,
                max=5,
                step=0.5,
                value=after_training[2],
                marks={i: str(i) for i in range(6)}
            ),
            html.Div(id='slider-output-container-2')
        ]),
        html.Div([
            dcc.Graph(id='indicator-3'),
            dcc.Slider(
                id='slider-3',
                min=0,
                max=5,
                step=0.5,
                value=after_training[3],
                marks={i: str(i) for i in range(6)}
            ),
            html.Div(id='slider-output-container-3')
        ]),
        html.Div([
            dcc.Graph(id='indicator-4'),
            dcc.Slider(
                id='slider-4',
                min=0,
                max=5,
                step=0.5,
                value=after_training[4],
                marks={i: str(i) for i in range(6)}
            ),
            html.Div(id='slider-output-container-4')
        ]),
        html.Div([
            dcc.Graph(id='indicator-5'),
            dcc.Slider(
                id='slider-5',
                min=0,
                max=5,
                step=0.5,
                value=after_training[5],
                marks={i: str(i) for i in range(6)}
            ),
            html.Div(id='slider-output-container-5')
        ]),
        html.Div([
            dcc.Graph(id='indicator-6'),
            dcc.Slider(
                id='slider-6',
                min=0,
                max=5,
                step=0.5,
                value=after_training[6],
                marks={i: str(i) for i in range(6)}
            ),
            html.Div(id='slider-output-container-6')
        ]),
        html.Div([
            dcc.Graph(id='indicator-7'),
            dcc.Slider(
                id='slider-7',
                min=0,
                max=5,
                step=0.5,
                value=after_training[7],
                marks={i: str(i) for i in range(6)}
            ),
            html.Div(id='slider-output-container-7')
        ])
    ])
])

@app.callback(
    [Output('radar-chart', 'figure')] +
    [Output(f'indicator-{i}', 'figure') for i in range(len(skills))] +
    [Output(f'slider-output-container-{i}', 'children') for i in range(len(skills))],
    [Input(f'slider-{i}', 'value') for i in range(len(skills))]
)
def update_charts(*slider_values):
    updated_after_training = list(slider_values)
    print(f"Slider values: {updated_after_training}")  # Debugging print

    # Create radar chart
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=before_training,
        theta=skills,
        fill='toself',
        name='Before Training'
    ))
    radar_fig.add_trace(go.Scatterpolar(
        r=updated_after_training,
        theta=skills,
        fill='toself',
        name='After Training'
    ))
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        title="Skill Levels Before and After Cyclopt Training"
    )

    # Create radial progress indicators
    indicators = []
    for i, (skill, before, after) in enumerate(zip(skills, before_training, updated_after_training)):
        indicators.append(go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=after,
            delta={'reference': before, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 5]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, before], 'color': "lightgray"},
                    {'range': [before, 5], 'color': "lightgreen"}],
            },
            title={'text': skill}
        )))

    output_texts = [f'Skill level for {skills[i]}: {value}' for i, value in enumerate(updated_after_training)]
    print("Returning outputs")  # Debugging print

    return [radar_fig] + indicators + output_texts

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
