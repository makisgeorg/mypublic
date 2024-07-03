import streamlit as st
import plotly.graph_objects as go

# Sample data for skill levels before and after training
skills = ['SQL Injection', 'XSS', 'CSRF', 'Authentication', 'Authorization', 'Encryption', 'Logging', 'Monitoring']
before_training = [3, 2, 4, 3, 2, 3, 2, 3]
after_training = [5, 4, 5, 4, 4, 5, 4, 5]

# Initialize the Streamlit app
st.title("Cyclopt Integration: Progress Tracking Dashboard")

# Create initial radar chart
def create_radar_chart(after_training):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=before_training,
        theta=skills,
        fill='toself',
        name='Before Integration'
    ))
    fig.add_trace(go.Scatterpolar(
        r=after_training,
        theta=skills,
        fill='toself',
        name='After Integration'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        title="Skill Levels Before and After Cyclopt Integration"
    )
    return fig

# Display the radar chart initially
radar_chart = st.plotly_chart(create_radar_chart(after_training))

# Create radial progress indicators (gauges) and sliders
for i, (skill, before, after) in enumerate(zip(skills, before_training, after_training)):
    st.subheader(f'{skill} Level')
    after_training[i] = st.slider(f'{skill}', 0, 5, after)
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=after_training[i],
        delta={'reference': before, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 5]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, before], 'color': "lightgray"},
                {'range': [before, 5], 'color': "lightgreen"}],
        },
        title={'text': skill}
    ))
    st.plotly_chart(gauge_fig)

# Update the radar chart based on the new slider values
radar_chart.plotly_chart(create_radar_chart(after_training))
