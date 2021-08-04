import streamlit as st
import yaml

from models import predict_model


def visualize():
    st.write('# Summarization  UI')
    st.markdown(
