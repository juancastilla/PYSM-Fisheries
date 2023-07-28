from utilities import *

st.set_page_config(page_title='Domains', layout="wide", page_icon="🔠")


#######################################
############ INIT STATE ###############
#######################################

if 'df_domains' not in st.session_state:
    sheet_id = '1YDsfTegWgBnH4KB4FJ5g7fegaaKFFItTkuOF3Fbc3Vc'
    sheet_name = 'DOMAINS'
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    df_domains=pd.read_csv(url)
    df_domains = df_domains.loc[:, ~df_domains.columns.str.contains('^Unnamed')]
    st.session_state.df_domains = df_domains

    
################################################################################################################################
######################################################## SIDEBAR ###############################################################
################################################################################################################################

#######################################
########### LOAD DOMAINS ##############
#######################################

st.sidebar.header("Load domains")

with st.sidebar.expander("Load domains from Google Sheets").form("load_domains_gsheet"):
    
    sheet_id = st.text_input('Google Sheets ID', '1YDsfTegWgBnH4KB4FJ5g7fegaaKFFItTkuOF3Fbc3Vc')
    press_load = st.form_submit_button("Load")
    
    if press_load:
        load_domains(sheet_id)

with st.sidebar.expander("Load domains from CSV file"):
    
    uploaded_file = st.file_uploader("Choose a domains file")
    
    if uploaded_file is not None:
        df_domains=pd.read_csv(uploaded_file)
        df_domains = df_domains.loc[:, ~df_domains.columns.str.contains('^Unnamed')]
        st.session_state.df_domains = df_domains

#######################################
########### ADD DOMAIN ################
#######################################

st.sidebar.header("Add domain")

with st.sidebar.expander("Add domain").form("add_domain"):
    domain_name = st.text_input('Name')
    domain_colour = st.color_picker('What colour would you like to use to categorise this new domain?', '#74d1ea')
    domain_other = st.text_input('Additional info')
        
    press_add = st.form_submit_button("Add")
    
    if press_add:
        add_domain(domain_name, domain_colour, domain_other)

#######################################
########### EDIT DOMAIN ###############
#######################################

st.sidebar.header("Edit domain")

with st.sidebar.expander("Edit domain"):

    domain_name_old = st.selectbox('Which domain would you like to edit?', st.session_state.df_domains.name)

    with st.sidebar.expander("Edit domain properties").form("edit_domain"):

        if domain_name_old == None:
            st.warning('there are no domains to edit')

        if domain_name_old != None:
            index_to_edit = st.session_state.df_domains[st.session_state.df_domains['name'] == domain_name_old].index.values[0]
        
            #preselected values
            domain_name_preselect = st.session_state.df_domains.at[index_to_edit,'name']
            domain_colour_preselect = "#000"
            domain_other_preselect = st.session_state.df_domains.at[index_to_edit,'other_info']

            domain_name_new = st.text_input('Name', value=domain_name_preselect)
            domain_colour_new = st.color_picker('What colour would you like to use to categorise this domain?', value=domain_colour_preselect)
            domain_other_new = st.text_input('Additional info', value=domain_other_preselect)

        press_edit = st.form_submit_button("Edit")

        if press_edit and domain_name_old != None:
            edit_domain(index_to_edit, domain_name_new, domain_colour_new, domain_other_new)

##########################################
############# DELETE DOMAIN ##############
##########################################

st.sidebar.header("Delete domain")

with st.sidebar.expander("Delete domain").form("delete_domain"):
    
    domain_name_to_delete = st.selectbox('Which domain would you like to delete?', st.session_state.df_domains.name)
    
    if domain_name_to_delete != None:
        index_to_delete = st.session_state.df_domains[st.session_state.df_domains['name'] == domain_name_to_delete].index.values[0]
    
    press_delete = st.form_submit_button("Delete")
    
    if press_delete and domain_name_to_delete != None:
        delete_domain(index_to_delete)
        st.experimental_rerun()

##########################################
############## DELETE ALL ################
##########################################

st.sidebar.header("Delete all")
with st.sidebar.expander("Delete all domains").form("delete_all_domains"):
    
    option = st.selectbox('Are you sure?', ('Yes', 'No'))
    
    if st.form_submit_button("Submit"):
        if option == 'Yes':
            sheet_id = '1YDsfTegWgBnH4KB4FJ5g7fegaaKFFItTkuOF3Fbc3Vc'
            sheet_name = 'DOMAINS'
            url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
            df_domains=pd.read_csv(url)
            df_domains = df_domains.loc[:, ~df_domains.columns.str.contains('^Unnamed')]
            st.session_state.df_domains = df_domains
            st.experimental_rerun()
        if option == 'No':
            st.experimental_rerun()

##########################################
############## SAVE ALL ##################
##########################################
            
st.sidebar.header("Save all")

csv = convert_df(st.session_state.df_domains)

st.sidebar.download_button(
     label="Download domains table as CSV",
     data=csv,
     file_name='domains.csv',
     mime='text/csv',
 )

################################################################################################################################
######################################################## MAIN PAGE #############################################################
################################################################################################################################

st.markdown('### Domains Table')
placeholder_domains_table = st.empty()
with placeholder_domains_table.expander("", expanded=True):
    st.session_state.df_domains['domain_id'] = st.session_state.df_domains.reset_index().index
    AgGrid(
        st.session_state.df_domains,
        theme="streamlit",
        fit_columns_on_grid_load=False)

st.markdown('### Domains Plot')
placeholder_domains_plot = st.empty()
with placeholder_domains_plot.expander("", expanded=True):    
    plot_domains()