import streamlit as st
import platform
from pathlib import Path

### PATH CONFIGURATION ###

if platform.system() == 'Darwin':
    main_path = Path(".")
else:
    main_path = Path("./streamlit")

### APP CONFIGURATION ###

apptitle = 'PYSM: Participatory Systems Mapping Workbench'

st.set_page_config(page_title=apptitle, layout="wide", page_icon=":house:")

### APP MAIN ###
st.sidebar.header('Settings')
app_language = st.sidebar.radio('Choose a language:', ('English', 'Spanish'), horizontal = True)
app_mode = st.sidebar.radio('Choose a use case:', ('Analysis', 'Workshop'), horizontal = True)
breakout_room = st.sidebar.radio('Choose a breakout room:', ('1', '2'), horizontal = True)

st.session_state.language = app_language
st.session_state.app_mode = app_mode
st.session_state.breakout_room = breakout_room

st.sidebar.markdown('#')

st.sidebar.image(str(main_path.joinpath('pysm.png')), width=300)

with st.sidebar:
    st.write("Streamlit version:", st.__version__)

from utilities import *