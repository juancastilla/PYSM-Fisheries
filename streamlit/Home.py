import streamlit as st
import platform
from pathlib import Path
from utilities import *

### PATH CONFIGURATION ###

if platform.system() == 'Darwin':
    main_path = Path(".")
else:
    main_path = Path("./streamlit")

### APP CONFIGURATION ###

apptitle = 'PYSM: Participatory Systems Mapping Workbench'

st.set_page_config(page_title=apptitle, layout="wide", page_icon=":house:")

### APP MAIN PAGE ###

landing_md = read_markdown_file("landing.md")
st.markdown(landing_md, unsafe_allow_html=True)

q1, q2, q3, q4, q5 = st.columns(5, gap="medium")

with q1:
   st.markdown("### Question 1")
   st.divider()
   st.markdown("What are the threats (i.e., what are the main problems?) to the sustainability of the focal fisheries in the Humboldt Current marine ecosystem, and how do they differ across the fisheries (southern hake, octopus, jumbo flying squid, and anchovy)?")

with q2:
   st.markdown("### Question 2")
   st.divider()
   st.markdown("Across the focal fisheries’ systems, what are the key elements and human behaviors that are prohibiting a transition to more sustainable fisheries?")

with q3:
   st.markdown("### Question 3")
   st.divider()
   st.markdown("What are the  system elements that could be targeted with interventions that have the most leverage to improve the sustainability of the focal fisheries?")

with q4:
   st.markdown("### Question 4")
   st.divider()
   st.markdown("How effective are the established methodologies focused on systems thinking and dynamics in identifying the landscape of threats, actors, and leverage points for change within fishery systems?")

with q5:
   st.markdown("### Question 5")
   st.divider()
   st.markdown("How does the data collected via surveys, interviews, and workshops contribute to the understanding of fishery systems and program design for improving fisheries sustainability?")

### APP SIDEBAR ###
st.sidebar.header('Settings')

app_language = st.sidebar.radio('Choose a language:', ('English', 'Spanish'), horizontal = True)
st.session_state.language = app_language

with st.sidebar.form(key='case_study_form'):

    fishery_selection = st.selectbox('Choose a Fishery:', ('Octopus Chile', 'Octopus Peru', 'Southern Hake', 'Jumbo Flying Squid', 'Anchoveta'))
    st.session_state.fishery = fishery_selection

    if fishery_selection == 'Octopus Chile':

        st.session_state.sheet_id = '1KyvP07oU4zuGlLQ61W12bSDDKyEtyFRJIthEPk0Iito'
    
    if fishery_selection == 'Octopus Peru':
    
        st.session_state.sheet_id = '1quqkUmq5BSf7i6Iv8L-KNFYTJddRHcux-kyK0ghpLaw'
    
    if fishery_selection == 'Southern Hake':
    
        st.session_state.sheet_id = '1m5NoPq_5TSH_FU32VNjVc5-uCfouuypydKoqbMHzwx4'
    
    if fishery_selection == 'Jumbo Flying Squid':

        st.session_state.sheet_id = '1lv8kJ67fmLV34qMYtijbUoMyKkyhXVxk0Hw92mNIAiI'
    
    if fishery_selection == 'Anchoveta':

        st.session_state.sheet_id = ''

    # Every form must have a submit button.
    submitted = st.form_submit_button("Load Case Study", type="primary")
   
    if submitted:
        pass

with st.sidebar:
    st.write("Streamlit version:", st.__version__)
    st.write("Pandas version:", pd.__version__)
    st.write("Seaborn version:", sns.__version__)
    st.write("Matplotlib version:", matplotlib.__version__)
    st.write("Python version:", platform.python_version())



st.sidebar.image(str(main_path.joinpath('pysm.png')), use_column_width='always')



##