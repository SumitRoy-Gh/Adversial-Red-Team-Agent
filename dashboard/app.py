# Run with: streamlit run dashboard/app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from memory.elo_tracker   import ELOTracker
from memory.exploit_store import ExploitStore

st.set_page_config(page_title='Red-Team Dashboard', layout='wide')
st.title('Adversarial Red-Team Agent — Live Dashboard')

tracker = ELOTracker()
store   = ExploitStore()

# Auto-refresh every 5 seconds
st.markdown('<meta http-equiv="refresh" content="5">', unsafe_allow_html=True)

# ── Metric cards ──────────────────────────────────────────────────
history = tracker.get_history()
df = pd.DataFrame(history)

col1, col2, col3, col4 = st.columns(4)
if len(df):
    last = df.iloc[-1]
    col1.metric('Total Rounds',      len(df))
    col2.metric('Attacker ELO',      f"{last['attacker_elo']:.0f}")
    col3.metric('Defender ELO',      f"{last['defender_elo']:.0f}")
    col4.metric('Total Exploits',    store.count())

# ── ELO capability curve ──────────────────────────────────────────
st.subheader('ELO Capability Curve')
if len(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['round_num'], y=df['attacker_elo'],
        name='Attacker ELO', line=dict(color='#E74C3C', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df['round_num'], y=df['defender_elo'],
        name='Defender ELO', line=dict(color='#2ECC71', width=2)
    ))
    fig.update_layout(
        xaxis_title='Round', yaxis_title='ELO Rating',
        height=400, hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Recent exploits table ─────────────────────────────────────────
st.subheader('Recent Exploits')
exploits = store.get_all_exploits()
if exploits:
    exploit_df = pd.DataFrame(exploits)
    exploit_df = exploit_df[['round_num','attack_type','severity','attack_text']]
    exploit_df['attack_text'] = exploit_df['attack_text'].str[:100] + '...'
    st.dataframe(exploit_df.sort_values('round_num', ascending=False).head(20),
                 use_container_width=True)
