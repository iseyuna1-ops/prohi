
import streamlit as st

page1 = st.Page(
    "overview.py",
    title="Dataset Overview",
    icon="📊"
)

page2 = st.Page(
    "diagnostic.py",
    title="Diagnostic Analysis",
    icon="🔬"
)

st.navigation([page1, page2]).run()