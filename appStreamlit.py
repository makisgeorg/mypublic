import streamlit as st
import plotly.graph_objects as go

# Sample data for skill levels before and after training
skills = ['SQL Injection', 'XSS', 'CSRF', 'Authentication', 'Authorization', 'Encryption', 'Logging', 'Monitoring']
before_training = [3, 2, 4, 3, 2, 3, 2, 3]
after_training = [5, 4, 5, 4, 4, 5, 4, 5]

st.title("Cyclopt Training Progress Dashboard")

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=before_training,
    theta=skills,
    fill='toself',
    name='Before Training'
))
fig.add_trace(go.Scatterpolar(
    r=after_training,
    theta=skills,
    fill='toself',
    name='After Training'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 5]
        )),
    title="Skill Levels Before and After Cyclopt Training"
)

st.plotly_chart(fig)

for i, skill in enumerate(skills):
    st.subheader(f'{skill} Level')
    after_training[i] = st.slider(f'{skill}', 0, 5, after_training[i])

fig.update_traces(r=after_training, selector=dict(name='After Training'))
st.plotly_chart(fig)
