from utilities import *

st.set_page_config(page_title='Factors', layout="wide", page_icon="🔠")

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

if 'df_factors' not in st.session_state:
    sheet_id = '1YDsfTegWgBnH4KB4FJ5g7fegaaKFFItTkuOF3Fbc3Vc'
    sheet_name = 'FACTORS'
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    df_factors=pd.read_csv(url)
    df_factors = df_factors.loc[:, ~df_factors.columns.str.contains('^Unnamed')]
    st.session_state.df_factors = df_factors

if 'df_relationships' not in st.session_state:
    sheet_id = '1YDsfTegWgBnH4KB4FJ5g7fegaaKFFItTkuOF3Fbc3Vc'
    sheet_name = 'RELATIONSHIPS'
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    df_rels=pd.read_csv(url)
    df_rels = df_rels.loc[:, ~df_rels.columns.str.contains('^Unnamed')]
    st.session_state.df_rels = df_rels

################################################################################################################################
######################################################## SIDEBAR ###############################################################
################################################################################################################################

st.sidebar.header("Load factors")

with st.sidebar.expander("Load factors from Google Sheets").form("load_factors_gsheet"):
    
    sheet_id = st.text_input('Google Sheets ID', '1YDsfTegWgBnH4KB4FJ5g7fegaaKFFItTkuOF3Fbc3Vc')
    st.warning('Warning: domains listed in the Factors Table will be appended to the ones already on the Domains Table. After importing the extrenal Factors Table from Google Sheets, please navigate back to Domains, review and consolidate the entries (make sure domains are unique). Then review/reassign/confirm domain allocations in the Factors Table using the Edit Factor tool below')
    
    press_load = st.form_submit_button("Load")
    
    if press_load:
        load_factors(sheet_id)
        add_domains_from_factors_table()

with st.sidebar.expander("Load factors from CSV file"):
    uploaded_file = st.file_uploader("Choose a factors file")
    if uploaded_file is not None:
        df_factors=pd.read_csv(uploaded_file)
        df_factors = df_factors.loc[:, ~df_factors.columns.str.contains('^Unnamed')]
        st.session_state.df_factors = df_factors
    
st.sidebar.header("Add factor")
with st.sidebar.expander("Add factor").form("add_factor"):
    
    factor_long_name =  st.text_input('Long name')
    factor_short_name =  st.text_input('Short name')
    factor_definition = st.text_input('How would you define this factor?')
    factor_domain = st.selectbox('To what domain does this factor belong?', st.session_state.df_domains.name)
    factor_importance = st.radio('How important is this factor?', ('Low', 'Medium', 'High', 'Not sure'), horizontal = True)
    factor_controllability = st.radio('How controllable is this factor?', ('Low', 'Medium', 'High', 'Not sure'), horizontal = True)
    factor_strategic = st.radio('How strategic is this factor?', ('Low', 'Medium', 'High', 'Not sure'), horizontal = True)
    factor_uncertainty = st.radio('How uncertain is this factor?', ('Low', 'Medium', 'High', 'Not sure'), horizontal = True)
    factor_vulnerable = st.radio('How vulnerable to change is this factor?', ('Low', 'Medium', 'High', 'Not sure'), horizontal = True)
    factor_resistant = st.radio('How resistant to change is this factor?', ('Low', 'Medium', 'High', 'Not sure'), horizontal = True)
    factor_info = st.text_input('Additional info')
        
    press_add = st.form_submit_button("Add")
    
    if press_add:
        add_factor(factor_long_name,
                   factor_short_name,
                   factor_definition,
                   factor_domain, 
                   factor_importance, 
                   factor_controllability, 
                   factor_strategic, 
                   factor_uncertainty, 
                   factor_vulnerable, 
                   factor_resistant, 
                   factor_info)
        
st.sidebar.header("Edit factor")

with st.sidebar.expander("Edit factor"):
    
    factor_name_old = st.selectbox('Which factor would you like to edit?', st.session_state.df_factors.long_name)

    with st.sidebar.expander("Edit factor properties").form("edit_factor"):

        if factor_name_old == None:
            st.warning('there are no factors to edit')

        if factor_name_old != None:
            index_to_edit = st.session_state.df_factors[st.session_state.df_factors['long_name'] == factor_name_old].index.values[0]

            #preselected values
            factor_long_name_preselect = st.session_state.df_factors.at[index_to_edit,'long_name']
            factor_short_name_preselect = st.session_state.df_factors.at[index_to_edit,'short_name']
            factor_definition_preselect = st.session_state.df_factors.at[index_to_edit,'definition']

            factor_domain_preselect = 0

            if st.session_state.df_factors.at[index_to_edit,'importance'] == 'Low':
                factor_importance_preselect = 0
            if st.session_state.df_factors.at[index_to_edit,'importance'] == 'Medium':
                factor_importance_preselect = 1
            if st.session_state.df_factors.at[index_to_edit,'importance'] == 'High':
                factor_importance_preselect = 2
            if st.session_state.df_factors.at[index_to_edit,'importance'] == 'Unclear':
                factor_importance_preselect = 3
            else: factor_importance_preselect = 3


            if st.session_state.df_factors.at[index_to_edit,'controllability'] == 'Low':
                factor_controllability_preselect = 0
            if st.session_state.df_factors.at[index_to_edit,'controllability'] == 'Medium':
                factor_controllability_preselect = 1
            if st.session_state.df_factors.at[index_to_edit,'controllability'] == 'High':
                factor_controllability_preselect = 2
            if st.session_state.df_factors.at[index_to_edit,'controllability'] == 'Unclear':
                factor_controllability_preselect = 3
            else: factor_controllability_preselect = 3

            if st.session_state.df_factors.at[index_to_edit,'strategic_importance'] == 'Low':
                factor_strategic_preselect = 0
            if st.session_state.df_factors.at[index_to_edit,'strategic_importance'] == 'Medium':
                factor_strategic_preselect = 1
            if st.session_state.df_factors.at[index_to_edit,'strategic_importance'] == 'High':
                factor_strategic_preselect = 2
            if st.session_state.df_factors.at[index_to_edit,'strategic_importance'] == 'Unclear':
                factor_strategic_preselect = 3
            else: factor_strategic_preselect = 3

            if st.session_state.df_factors.at[index_to_edit,'uncertainty'] == 'Low':
                factor_uncertainty_preselect = 0
            if st.session_state.df_factors.at[index_to_edit,'uncertainty'] == 'Medium':
                factor_uncertainty_preselect = 1
            if st.session_state.df_factors.at[index_to_edit,'uncertainty'] == 'High':
                factor_uncertainty_preselect = 2
            if st.session_state.df_factors.at[index_to_edit,'uncertainty'] == 'Unclear':
                factor_uncertainty_preselect = 3
            else: factor_uncertainty_preselect = 3

            if st.session_state.df_factors.at[index_to_edit,'vulnerable_to_change'] == 'Low':
                factor_vulnerable_preselect = 0
            if st.session_state.df_factors.at[index_to_edit,'vulnerable_to_change'] == 'Medium':
                factor_vulnerable_preselect = 1
            if st.session_state.df_factors.at[index_to_edit,'vulnerable_to_change'] == 'High':
                factor_vulnerable_preselect = 2
            if st.session_state.df_factors.at[index_to_edit,'vulnerable_to_change'] == 'Unclear':
                factor_vulnerable_preselect = 3
            else: factor_vulnerable_preselect = 3

            if st.session_state.df_factors.at[index_to_edit,'resistant_to_change'] == 'Low':
                factor_resistant_preselect = 0
            if st.session_state.df_factors.at[index_to_edit,'resistant_to_change'] == 'Medium':
                factor_resistant_preselect = 1
            if st.session_state.df_factors.at[index_to_edit,'resistant_to_change'] == 'High':
                factor_resistant_preselect = 2
            if st.session_state.df_factors.at[index_to_edit,'resistant_to_change'] == 'Unclear':
                factor_resistant_preselect = 3
            else: factor_resistant_preselect = 3

            factor_info_preselect = str(st.session_state.df_factors.at[index_to_edit,'other_info'])   

            factor_long_name_new =  st.text_input('Long name', value=factor_long_name_preselect)
            factor_short_name_new =  st.text_input('Short name', value=factor_short_name_preselect)
            factor_definition_new = st.text_input('How would you define this factor?', value=factor_definition_preselect)
            factor_domain_new = st.selectbox('To what domain does this factor belong?', st.session_state.df_domains.name, index=factor_domain_preselect)
            factor_importance_new = st.radio('How important is this factor?', ('Low', 'Medium', 'High', 'Unsure'), horizontal = True, index=factor_importance_preselect)
            factor_controllability_new = st.radio('How controllable is this factor?', ('Low', 'Medium', 'High', 'Unsure'), horizontal = True, index=factor_controllability_preselect)
            factor_strategic_new = st.radio('How strategic is this factor?', ('Low', 'Medium', 'High', 'Unsure'), horizontal = True, index=factor_strategic_preselect)
            factor_uncertainty_new = st.radio('How uncertain is this factor?', ('Low', 'Medium', 'High', 'Unsure'), horizontal = True, index=factor_uncertainty_preselect)
            factor_vulnerable_new = st.radio('How vulnerable to change is this factor?', ('Low', 'Medium', 'High', 'Unsure'), horizontal = True, index=factor_vulnerable_preselect)
            factor_resistant_new = st.radio('How resistant to change is this factor?', ('Low', 'Medium', 'High', 'Unsure'), horizontal = True, index=factor_resistant_preselect)
            factor_info_new = st.text_input('Additional info', value=factor_info_preselect)

        press_edit = st.form_submit_button("Edit")

        if press_edit and factor_name_old != None:
            edit_factor(index_to_edit,
                factor_long_name_new, 
                factor_short_name_new, 
                factor_definition_new, 
                factor_domain_new, 
                factor_importance_new, 
                factor_controllability_new, 
                factor_strategic_new, 
                factor_uncertainty_new,
                factor_vulnerable_new,
                factor_resistant_new,
                factor_info_new
               )
    
st.sidebar.header("Delete factor")
with st.sidebar.expander("Delete factor").form("delete_factor"):
    factor_name_to_delete = st.selectbox('Which factor would you like to delete?', st.session_state.df_factors.long_name)
    if factor_name_to_delete != None:
        index_to_delete = st.session_state.df_factors[st.session_state.df_factors['long_name'] == factor_name_to_delete].index.values[0]
    
    press_delete = st.form_submit_button("Delete")
    
    if press_delete and factor_name_to_delete != None:
        delete_factor(index_to_delete)
        st.experimental_rerun()
        
st.sidebar.header("Delete all")
with st.sidebar.expander("Delete all factors").form("delete_all_factros"):
    
    option = st.selectbox('Are you sure?', ('Yes', 'No'))
    if st.form_submit_button("Submit"):
        if option == 'Yes':
            sheet_id = '1YDsfTegWgBnH4KB4FJ5g7fegaaKFFItTkuOF3Fbc3Vc'
            sheet_name = 'FACTORS'
            url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
            df_factors=pd.read_csv(url)
            df_factors = df_factors.loc[:, ~df_factors.columns.str.contains('^Unnamed')]
            st.session_state.df_factors = df_factors
            st.experimental_rerun()
        if option == 'No':
            st.experimental_rerun()

st.sidebar.header("Save all")

csv = convert_df(st.session_state.df_factors)

st.sidebar.download_button(
     label="Download factors table as CSV",
     data=csv,
     file_name='factors.csv',
     mime='text/csv',
 )


################################################################################################################################
######################################################## MAIN PAGE #############################################################
################################################################################################################################

st.markdown('### Factors Table')

placeholder_factors_table = st.empty()


with placeholder_factors_table.expander("", expanded=True):
    
    st.session_state.df_factors['factor_id'] = st.session_state.df_factors.reset_index().index
    
    AgGrid(st.session_state.df_factors)

st.markdown('### Factors Plot')

placeholder_factors_plot = st.empty()

with placeholder_factors_plot.expander("", expanded=True):        
    plot_factors()
    
    st.markdown('### General and Focal Factors')
    HtmlFile = open('G_factors.html','r',encoding='utf-8')
    components.html(HtmlFile.read(),height=1000)
