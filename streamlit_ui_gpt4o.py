# streamlit_ui.py

import streamlit as st
from retrival_fn import check_requirement_validity

st.set_page_config(page_title="INCOSE Requirement Validator", layout="centered")

# UI Title
st.title("📋 INCOSE Requirement Validator (GPT-4o)")
st.markdown("Enter a system requirement below. It will be validated using INCOSE standards and GPT-4o reasoning.")

# Input text field
requirement_input = st.text_area("✍️ Enter your requirement to validate")

# On button click
if st.button("✅ Check"):
    if not requirement_input.strip():
        st.warning("Please enter a requirement first.")
    else:
        with st.spinner("🔍 Validating using GPT-4o and INCOSE..."):
            result = check_requirement_validity(requirement_input)
        st.success("✅ Validation complete")
        st.markdown(result)
