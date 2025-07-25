# streamlit_ui_groq.py

import streamlit as st
from retrival_fn_groq import check_requirement_validity

st.set_page_config(page_title="INCOSE Requirement Validator (Groq)", layout="centered")

st.title("📋 INCOSE Requirement Validator (Groq)")
st.markdown("Validate system requirements based on INCOSE standards using open-source LLMs (Groq).")

# User input
query = st.text_area("✍️ Enter your system requirement:")

# Model options
model_map = {
    "LLaMA 3 - 70B": "llama3-70b-8192",
    "Kimi K2": "moonshotai/kimi-k2-instruct",
    "Gemma 2": "gemma2-9b-it"
}

# Store selected model in session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
    st.session_state.selected_model_label = None

# Model selection buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🦙 LLaMA 3 - 70B"):
        st.session_state.selected_model = model_map["LLaMA 3 - 70B"]
        st.session_state.selected_model_label = "LLaMA 3 - 70B"
with col2:
    if st.button("🧠 Kimi K2"):
        st.session_state.selected_model = model_map["Kimi K2"]
        st.session_state.selected_model_label = "Kimi K2"
with col3:
    if st.button("🤖 Gemma 2"):
        st.session_state.selected_model = model_map["Gemma 2"]
        st.session_state.selected_model_label = "Gemma 2"

# Validation button
if st.button("✅ Check Requirement"):
    if not query.strip():
        st.warning("Please enter a requirement first.")
    elif not st.session_state.selected_model:
        st.warning("Please select a model before validation.")
    else:
        with st.spinner(f"Analyzing using {st.session_state.selected_model_label}..."):
            result = check_requirement_validity(query, st.session_state.selected_model)
        st.success(f"Model Used: {st.session_state.selected_model_label}")
        st.markdown(result)
