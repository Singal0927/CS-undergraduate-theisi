import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling

st.header('疫情数据的profiling')
profiling=st.session_state['data'].profile_report()
st_profile_report(profiling)