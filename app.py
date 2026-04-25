import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# PAGE CONFIG + DARK THEME
# -------------------------------
st.set_page_config(page_title="IPL Live AI", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0b0c10;
    color: white;
}
.stApp {
    background-color: #0b0c10;
}
.card {
    background: #1f2833;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
.title {
    font-size: 28px;
    font-weight: bold;
    color: #66fcf1;
}
.team {
    font-size: 22px;
    font-weight: bold;
}
.score {
    font-size: 36px;
    color: #45a29e;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🏏 IPL LIVE AI DASHBOARD</h1>", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    matches = pd.read_csv(
        "https://raw.githubusercontent.com/riteshkc/ipl-data/master/data/matches.csv"
    )
    deliveries = pd.read_csv(
        "https://raw.githubusercontent.com/riteshkc/ipl-data/master/data/deliveries.csv"
    )
    return matches, deliveries

matches, deliveries = load_data()

# -------------------------------
# TRAIN MODEL
# -------------------------------
@st.cache_resource
def train_model():
    X = [
        [50, 30, 5],
        [20, 10, 2],
        [80, 50, 7],
        [10, 5, 1],
        [60, 40, 6],
        [30, 20, 4]
    ]
    y = [1, 0, 1, 0, 1, 0]

    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model()

# -------------------------------
# TEAM SELECTOR
# -------------------------------
teams = sorted(matches['team1'].dropna().unique())

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Batting Team", teams)

with col2:
    team2 = st.selectbox("Bowling Team", teams)

# -------------------------------
# MATCH INPUT (HOTSTAR STYLE)
# -------------------------------
st.markdown("### 🎮 Live Match Controls")

col1, col2, col3, col4 = st.columns(4)

with col1:
    target = st.number_input("Target", 100, 300, 180)

with col2:
    score = st.number_input("Score", 0, 300, 100)

with col3:
    overs = st.slider("Overs", 0, 20, 10)

with col4:
    wickets = st.slider("Wickets", 0, 10, 7)

# -------------------------------
# CALCULATIONS
# -------------------------------
runs_left = max(target - score, 0)
balls_left = max(120 - (overs * 6), 1)

required_rr = (runs_left / balls_left) * 6

input_data = pd.DataFrame(
    [[runs_left, balls_left, wickets]],
    columns=["runs_left", "balls_left", "wickets"]
)

prob = model.predict_proba(input_data)[0][1]

# Smart adjustment
if required_rr > 12:
    prob *= 0.5
elif required_rr > 9:
    prob *= 0.7
elif wickets <= 2:
    prob *= 0.6

prob = max(0, min(prob, 1))

# -------------------------------
# 🔥 HOTSTAR SCOREBOARD
# -------------------------------
st.markdown("### 🏟️ Live Scoreboard")

col1, col2, col3 = st.columns([3,1,3])

with col1:
    st.markdown(f"""
    <div class='card'>
        <div class='team'>{team1}</div>
        <div class='score'>{score}/{10-wickets}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<h2 style='text-align:center;'>VS</h2>", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='card'>
        <div class='team'>{team2}</div>
        <div class='score'>Target: {target}</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# 🎯 WIN PROBABILITY
# -------------------------------
st.markdown("### 🤖 Win Probability")

st.progress(prob)
st.success(f"{round(prob*100,2)}% chance to win")

# -------------------------------
# 📊 MATCH STATS
# -------------------------------
colA, colB, colC = st.columns(3)

with colA:
    st.metric("Runs Left", runs_left)

with colB:
    st.metric("Balls Left", balls_left)

with colC:
    st.metric("Req RR", round(required_rr,2))

# -------------------------------
# 📈 GRAPH
# -------------------------------
st.markdown("### 📊 Match Analysis")

chart_data = pd.DataFrame({
    "Metric": ["Runs Left", "Balls Left", "Wickets"],
    "Value": [runs_left, balls_left, wickets]
})

st.bar_chart(chart_data.set_index("Metric"))
