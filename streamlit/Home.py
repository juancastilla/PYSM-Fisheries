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

st.markdown("# Welcome to the Collaborative Systems Mapping Workbench (CSM Workbench)")
st.markdown("## An open-source tool to enhance management effectiveness, monitoring, evaluation and learning in complex socio-environmental systems")

# Display the landing image
landing_image_path = main_path / "landing_gpt.png"
if landing_image_path.exists():
    st.image(str(landing_image_path), use_container_width=True)
else:
    st.error("Landing image not found. Please ensure 'landing_gpt.png' is in the correct directory.")



# landing_md = read_markdown_file("landing.md")
# st.markdown(landing_md, unsafe_allow_html=True)

# q1, q2, q3, q4, q5 = st.columns(5, gap="medium")

# with q1:
#    st.markdown("### Question 1")
#    st.divider()
#    st.markdown("What are the threats (i.e., what are the main problems?) to the sustainability of the focal fisheries in the Humboldt Current marine ecosystem, and how do they differ across the fisheries (southern hake, octopus, jumbo flying squid, and anchovy)?")

# with q2:
#    st.markdown("### Question 2")
#    st.divider()
#    st.markdown("Across the focal fisheries’ systems, what are the key elements and human behaviors that are prohibiting a transition to more sustainable fisheries?")

# with q3:
#    st.markdown("### Question 3")
#    st.divider()
#    st.markdown("What are the  system elements that could be targeted with interventions that have the most leverage to improve the sustainability of the focal fisheries?")

# with q4:
#    st.markdown("### Question 4")
#    st.divider()
#    st.markdown("How effective are the established methodologies focused on systems thinking and dynamics in identifying the landscape of threats, actors, and leverage points for change within fishery systems?")

# with q5:
#    st.markdown("### Question 5")
#    st.divider()
#    st.markdown("How does the data collected via surveys, interviews, and workshops contribute to the understanding of fishery systems and program design for improving fisheries sustainability?")

### APP SIDEBAR ###
st.sidebar.header('Settings')

app_language = st.sidebar.radio('Choose a language:', ('English', 'Spanish'), horizontal = True)
st.session_state.language = app_language

with st.sidebar.form(key='case_study_form'):

    fishery_selection = st.selectbox('Choose a Case Study:', ('Octopus Chile', 'Octopus Peru', 'Southern Hake', 'Jumbo Flying Squid', 'Anchoveta', 'Marine Litter', 'Octopus Reduced', 'Coastal Basins'))
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

        st.session_state.sheet_id = '1OnuDvh1RFL8XVcif819htafTgrCRhpzGzJxd9pYXqUQ'
    
    if fishery_selection == 'Marine Litter':

        st.session_state.sheet_id = '1GaFkxjtJrgnBQk41_n_I5HT2km3v_LAhWtVYT0eExo8'

    if fishery_selection == 'Octopus Reduced':

        st.session_state.sheet_id = '1XEd9LZmfb-nXoV9nT_uIIQWgCwih8mTOrZAKBQXBECQ'

    if fishery_selection == 'Coastal Basins':

        st.session_state.sheet_id = '1Hb6ICM1-RhezF3ilTOStz_u00H49_mTsfGk3IbCoJbo'

    # Every form must have a submit button.
    submitted = st.form_submit_button("Load Case Study", type="primary")
   
    if submitted:
        pass

# with st.sidebar:
#     st.write("OS:", platform.system())
#     st.write("Python version:", platform.python_version())
#     st.write("Streamlit version:", st.__version__)
#     st.write("Numpy version:", np.__version__)
#     st.write("Pandas version:", pd.__version__)
#     st.write("Matplotlib version:", matplotlib.__version__)
#     st.write("Seaborn version:", sns.__version__)


st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
    
st.sidebar.image(str(main_path.joinpath('pysm.png')), width=300)



##