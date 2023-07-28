from utilities import *

st.set_page_config(page_title='Workshop 2 (Validation)', layout="wide", page_icon="🔬")

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
    df_relationships=pd.read_csv(url)
    df_relationships = df_relationships.loc[:, ~df_relationships.columns.str.contains('^Unnamed')]
    st.session_state.df_relationships = df_relationships

################################################################################################################################
######################################################## SIDEBAR ###############################################################
################################################################################################################################

#### LOAD DATA (AUTOMATIC ON STARTUP) ####

# Verification workshop mode: Consolidated map factors and relationships are automatically loaded
# sheet_id = '1YDsfTegWgBnH4KB4FJ5g7fegaaKFFItTkuOF3Fbc3Vc'
# load_factors(sheet_id)
# load_relationships(sheet_id)
# load_domains(sheet_id)

#### LOAD DATA (MANUAL) ####

### Original data ###
if st.sidebar.button('Load Data'):
    sheet_id = '1YDsfTegWgBnH4KB4FJ5g7fegaaKFFItTkuOF3Fbc3Vc'
    load_factors(sheet_id)
    load_relationships(sheet_id)
    load_domains(sheet_id)
    st.session_state.domains_counter = 6
    st.session_state.factors_counter = 50
    st.session_state.relationships_counter = 155
    
if platform.system() == 'Darwin':
    filename='service_account.json'
else:
    filename='streamlit/service_account.json'

gc = gspread.service_account(filename=filename)
sh = gc.open("PYSM templates")

if st.session_state.app_mode == 'Workshop' and st.session_state.breakout_room == '1':
    st.session_state.domains_backup_wks = sh.worksheet('DOM_BACKUP_1')
    st.session_state.factors_backup_wks = sh.worksheet('FAC_BACKUP_1')
    st.session_state.relationships_backup_wks = sh.worksheet('REL_BACKUP_1')
    
if st.session_state.app_mode == 'Workshop' and st.session_state.breakout_room == '2':
    st.session_state.domains_backup_wks = sh.worksheet('DOM_BACKUP_2')
    st.session_state.factors_backup_wks = sh.worksheet('FAC_BACKUP_2')
    st.session_state.relationships_backup_wks = sh.worksheet('REL_BACKUP_2')
    
if st.session_state.app_mode == 'Analysis':
    st.session_state.domains_backup_wks = sh.worksheet('DOM_BACKUP')
    st.session_state.factors_backup_wks = sh.worksheet('FAC_BACKUP')
    st.session_state.relationships_backup_wks = sh.worksheet('REL_BACKUP')
    
### Load backup data
if st.sidebar.button('Load Backup'):
    load_backup()
    
### Save backup data
if st.sidebar.button('Save Backup'):
    save_backup()

    
######## COUNTERS #######


    
######## DOMAINS ###########
st.sidebar.header("DOMAINS")

st.sidebar.header("Add domain")

with st.sidebar.expander("Add domain").form("add_domain"):
    domain_name = st.text_input('Name')
    domain_colour = st.color_picker('What colour would you like to use to categorise this new domain?', '#74d1ea')
    domain_other = st.text_input('Additional info')
        
    press_add = st.form_submit_button("Add")
    
    if press_add:
        add_domain(domain_name, domain_colour, domain_other)
        save_backup()
        
st.sidebar.header("Edit domain")

with st.sidebar.expander("Edit domain"):

    domain_name_old = st.selectbox('Which domain would you like to edit?', st.session_state.df_domains.domain_name)

    with st.sidebar.expander("Edit domain properties").form("edit_domain"):

        if domain_name_old == None:
            st.warning('there are no domains to edit')

        if domain_name_old != None:
            index_to_edit = st.session_state.df_domains[st.session_state.df_domains['domain_name'] == domain_name_old].index.values[0]
        
            #preselected values
            domain_name_preselect = st.session_state.df_domains.at[index_to_edit,'domain_name']
            domain_colour_preselect = "#000"
            domain_other_preselect = st.session_state.df_domains.at[index_to_edit,'domain_info']

            domain_name_new = st.text_input('Name', value=domain_name_preselect)
            domain_colour_new = st.color_picker('What colour would you like to use to categorise this domain?', value=domain_colour_preselect)
            domain_other_new = st.text_input('Additional info', value=domain_other_preselect)

        press_edit = st.form_submit_button("Edit")

        if press_edit and domain_name_old != None:
            edit_domain(index_to_edit, domain_name_new, domain_colour_new, domain_other_new)
            save_backup()

st.sidebar.header("Delete domain")

with st.sidebar.expander("Delete domain").form("delete_domain"):
    
    domain_name_to_delete = st.selectbox('Which domain would you like to delete?', st.session_state.df_domains.domain_name)
    
    if domain_name_to_delete != None:
        index_to_delete = st.session_state.df_domains[st.session_state.df_domains['domain_name'] == domain_name_to_delete].index.values[0]
    
    press_delete = st.form_submit_button("Delete")
    
    if press_delete and domain_name_to_delete != None:
        delete_domain(index_to_delete)
        save_backup()
        st.experimental_rerun()
        

with st.sidebar:
    st.markdown("""---""")


######## FACTORS ###########
st.sidebar.header("FACTORS")

st.sidebar.header("add factor")

with st.sidebar.expander("add factor").form("add_factor"):
    
    
    factor_domain = st.selectbox('To what domain does this factor belong?', st.session_state.df_domains.domain_name)
    
    factor_long_name =  st.text_input('Long name')
    factor_short_name =  st.text_input('Short name')
    factor_definition = st.text_input('How would you define this factor?')
    
    factor_importance = st.radio('How important is this factor?', ('Not sure','Low', 'Medium', 'High'), horizontal = True)
    factor_controllability = st.radio('How controllable is this factor?', ('Not sure','Low', 'Medium', 'High'), horizontal = True)
    factor_strategic = st.radio('How strategic is this factor?', ('Not sure','Low', 'Medium', 'High'), horizontal = True)
    factor_uncertainty = st.radio('How uncertain is this factor?', ('Not sure','Low', 'Medium', 'High'), horizontal = True)
    factor_vulnerable = st.radio('How vulnerable to change is this factor?', ('Not sure','Low', 'Medium', 'High'), horizontal = True)
    factor_resistant = st.radio('How resistant to change is this factor?', ('Not sure','Low', 'Medium', 'High'), horizontal = True)
    factor_info = st.text_input('Additional info')
        
    press_add = st.form_submit_button("add")
    
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
        save_backup()
        
st.sidebar.header("edit factor")

with st.sidebar.expander("edit factor"):
    
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
                factor_importance_preselect = 1
            if st.session_state.df_factors.at[index_to_edit,'importance'] == 'Medium':
                factor_importance_preselect = 2
            if st.session_state.df_factors.at[index_to_edit,'importance'] == 'High':
                factor_importance_preselect = 3
            if st.session_state.df_factors.at[index_to_edit,'importance'] == 'Not sure':
                factor_importance_preselect = 0
            else: factor_importance_preselect = 0


            if st.session_state.df_factors.at[index_to_edit,'controllability'] == 'Low':
                factor_controllability_preselect = 1
            if st.session_state.df_factors.at[index_to_edit,'controllability'] == 'Medium':
                factor_controllability_preselect = 2
            if st.session_state.df_factors.at[index_to_edit,'controllability'] == 'High':
                factor_controllability_preselect = 3
            if st.session_state.df_factors.at[index_to_edit,'controllability'] == 'Not sure':
                factor_controllability_preselect = 0
            else: factor_controllability_preselect = 0

            if st.session_state.df_factors.at[index_to_edit,'strategic_importance'] == 'Low':
                factor_strategic_preselect = 1
            if st.session_state.df_factors.at[index_to_edit,'strategic_importance'] == 'Medium':
                factor_strategic_preselect = 2
            if st.session_state.df_factors.at[index_to_edit,'strategic_importance'] == 'High':
                factor_strategic_preselect = 3
            if st.session_state.df_factors.at[index_to_edit,'strategic_importance'] == 'Not sure':
                factor_strategic_preselect = 0
            else: factor_strategic_preselect = 0

            if st.session_state.df_factors.at[index_to_edit,'uncertainty'] == 'Low':
                factor_uncertainty_preselect = 1
            if st.session_state.df_factors.at[index_to_edit,'uncertainty'] == 'Medium':
                factor_uncertainty_preselect = 2
            if st.session_state.df_factors.at[index_to_edit,'uncertainty'] == 'High':
                factor_uncertainty_preselect = 3
            if st.session_state.df_factors.at[index_to_edit,'uncertainty'] == 'Not sure':
                factor_uncertainty_preselect = 0
            else: factor_uncertainty_preselect = 0

            if st.session_state.df_factors.at[index_to_edit,'vulnerable_to_change'] == 'Low':
                factor_vulnerable_preselect = 1
            if st.session_state.df_factors.at[index_to_edit,'vulnerable_to_change'] == 'Medium':
                factor_vulnerable_preselect = 2
            if st.session_state.df_factors.at[index_to_edit,'vulnerable_to_change'] == 'High':
                factor_vulnerable_preselect = 3
            if st.session_state.df_factors.at[index_to_edit,'vulnerable_to_change'] == 'Not sure':
                factor_vulnerable_preselect = 0
            else: factor_vulnerable_preselect = 0

            if st.session_state.df_factors.at[index_to_edit,'resistant_to_change'] == 'Low':
                factor_resistant_preselect = 1
            if st.session_state.df_factors.at[index_to_edit,'resistant_to_change'] == 'Medium':
                factor_resistant_preselect = 2
            if st.session_state.df_factors.at[index_to_edit,'resistant_to_change'] == 'High':
                factor_resistant_preselect = 3
            if st.session_state.df_factors.at[index_to_edit,'resistant_to_change'] == 'Not sure':
                factor_resistant_preselect = 0
            else: factor_resistant_preselect = 0

            factor_info_preselect = str(st.session_state.df_factors.at[index_to_edit,'other_info'])   

            factor_long_name_new =  st.text_input('Long name', value=factor_long_name_preselect)
            factor_short_name_new =  st.text_input('Short name', value=factor_short_name_preselect)
            factor_definition_new = st.text_input('How would you define this factor?', value=factor_definition_preselect)
            factor_domain_new = st.selectbox('To what domain does this factor belong?', st.session_state.df_domains.domain_name, index=factor_domain_preselect)
            factor_importance_new = st.radio('How important is this factor?', ('Not sure', 'Low', 'Medium', 'High'), horizontal = True, index=factor_importance_preselect)
            factor_controllability_new = st.radio('How controllable is this factor?', ('Not sure','Low', 'Medium', 'High'), horizontal = True, index=factor_controllability_preselect)
            factor_strategic_new = st.radio('How strategic is this factor?', ('Not sure','Low', 'Medium', 'High'), horizontal = True, index=factor_strategic_preselect)
            factor_uncertainty_new = st.radio('How uncertain is this factor?', ('Not sure','Low', 'Medium', 'High'), horizontal = True, index=factor_uncertainty_preselect)
            factor_vulnerable_new = st.radio('How vulnerable to change is this factor?', ('Not sure','Low', 'Medium', 'High'), horizontal = True, index=factor_vulnerable_preselect)
            factor_resistant_new = st.radio('How resistant to change is this factor?', ('Not sure','Low', 'Medium', 'High'), horizontal = True, index=factor_resistant_preselect)
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
            save_backup()
    
st.sidebar.header("delete factor")

with st.sidebar.expander("delete factor").form("delete_factor"):
    
    factor_name_to_delete = st.selectbox('Which factor would you like to delete?', st.session_state.df_factors.long_name)
    
    if factor_name_to_delete != None:
        index_to_delete = st.session_state.df_factors[st.session_state.df_factors['long_name'] == factor_name_to_delete].index.values[0]
    
    press_delete = st.form_submit_button("delete")
    
    if press_delete and factor_name_to_delete != None:
        
        delete_factor(index_to_delete, factor_name_to_delete)
        save_backup()
        st.experimental_rerun()
              
st.sidebar.header("save factors")

csv = convert_df(st.session_state.df_factors)

st.sidebar.download_button(
     label="Download factors table as CSV",
     data=csv,
     file_name='factors.csv',
     mime='text/csv',
 )

with st.sidebar:
    st.markdown("""---""")

######## RELATIONSHIPS ###########

st.sidebar.header("RELATIONSHIPS")

## ADD ##
st.sidebar.header("add relationship")

with st.sidebar.expander("Add relationships").form("add_relationships"):
    
    relationship_from =  st.selectbox('What is the origin of this relationship?', st.session_state.df_factors.long_name, index=0)
    relationship_to =  st.selectbox('What is the destination of this relationship?', st.session_state.df_factors.long_name, index=0)
    relationship_polarity =  st.radio('What is the direction of is this relationship?', ('Positive', 'Negative', 'Unclear'), horizontal = True)
    relationship_strength =  st.radio('What is the strength of this relationship?', ('Low', 'Medium', 'High', 'Unclear'), horizontal = True)
    relationship_temporality =  st.radio('What is the temporality of this relationship?', ('Immediate', 'Slight Delay', 'Significant Delay', 'Unclear'), horizontal = True)
    relationship_importance = st.radio('How important is this relationship?', ('Low', 'Medium', 'High', 'Unclear'), horizontal = True)
        
    press_add = st.form_submit_button("Add")
    
    if relationship_from == relationship_to:
        check_to_from = True
    else:
        check_to_from = False
    
    if press_add and check_to_from == False:
        add_relationship(relationship_from,
                         relationship_to,
                         relationship_polarity, 
                         relationship_strength, 
                         relationship_temporality,
                         relationship_importance)
        save_backup()


## EDIT ##
        
    # st.sidebar.header("edit relationship")

    # with st.sidebar.expander("Edit relationship"):

    #     relationship_id_old = st.selectbox('Which relationship would you like to edit?', st.session_state.df_relationships.relationship_id)

    #     with st.sidebar.expander("Edit relationship properties").form("edit_relationship"):

    #         if relationship_id_old == None:
    #             st.warning('there are no relationships to edit')

    #         if relationship_id_old != None:
    #             index_to_edit = st.session_state.df_relationships[st.session_state.df_relationships['relationship_id'] == relationship_id_old].index.values[0]

    #             #preselected values

    #             relationship_from_preselect = 0
    #             relationship_to_preselect = 0
    #             relationship_polarity_preselect = 0
    #             relationship_strength_preselect = 0
    #             relationship_temporality_preselect = 0
    #             relationship_importance_preselect = 0

    #             if st.session_state.df_relationships.at[index_to_edit,'polarity'] == 'Positive':
    #                 relationship_polarity_preselect = 0
    #             if st.session_state.df_relationships.at[index_to_edit,'polarity'] == 'Negative':
    #                 relationship_polarity_preselect = 1
    #             if st.session_state.df_relationships.at[index_to_edit,'polarity'] == 'Unclear':
    #                 relationship_polarity_preselect = 2

    #             if st.session_state.df_relationships.at[index_to_edit,'strength'] == 'Low':
    #                 relationship_strength_preselect = 0
    #             if st.session_state.df_relationships.at[index_to_edit,'strength'] == 'Medium':
    #                 relationship_strength_preselect = 1
    #             if st.session_state.df_relationships.at[index_to_edit,'strength'] == 'High':
    #                 relationship_strength_preselect = 2
    #             if st.session_state.df_relationships.at[index_to_edit,'strength'] == 'Unclear':
    #                 relationship_strength_preselect = 3 

    #             if st.session_state.df_relationships.at[index_to_edit,'temporality'] == 'Immediate':
    #                 relationship_temporality_preselect = 0
    #             if st.session_state.df_relationships.at[index_to_edit,'temporality'] == 'Slight Delay':
    #                 relationship_temporality_preselect = 1
    #             if st.session_state.df_relationships.at[index_to_edit,'temporality'] == 'Significant Delay':
    #                 relationship_temporality_preselect = 2
    #             if st.session_state.df_relationships.at[index_to_edit,'temporality'] == 'Unclear':
    #                 relationship_temporality_preselect = 3        

    #             if st.session_state.df_relationships.at[index_to_edit,'importance'] == 'Low':
    #                 relationship_importance_preselect = 0
    #             if st.session_state.df_relationships.at[index_to_edit,'importance'] == 'Medium':
    #                 relationship_importance_preselect = 1
    #             if st.session_state.df_relationships.at[index_to_edit,'importance'] == 'High':
    #                 relationship_importance_preselect = 2
    #             if st.session_state.df_relationships.at[index_to_edit,'importance'] == 'Unclear':
    #                 relationship_importance_preselect = 3 

    #             relationship_from =  st.selectbox('What is the origin of this relationship?', st.session_state.df_factors.long_name, index=0)
    #             relationship_to =  st.selectbox('What is the destination of this relationship?', st.session_state.df_factors.long_name, index=0)
    #             relationship_polarity =  st.radio('What is the direction of is this relationship?', ('Positive', 'Negative', 'Unclear'), horizontal = True, index=relationship_polarity_preselect)
    #             relationship_strength =  st.radio('What is the strength of this relationship?', ('Low', 'Medium', 'High', 'Unclear'), horizontal = True, index=relationship_strength_preselect)
    #             relationship_temporality =  st.radio('What is the temporality of this relationship?', ('Immediate', 'Slight Delay', 'Significant Delay', 'Unclear'), horizontal = True, index=relationship_temporality_preselect)
    #             relationship_importance = st.radio('How important is this relationship?', ('Low', 'Medium', 'High', 'Unclear'), horizontal = True, index=relationship_importance_preselect)

    #         press_edit = st.form_submit_button("Edit")

    # #         if press_edit and factor_name_old != None:
    # #             edit_factor(index_to_edit,
    # #                 factor_long_name_new, 
    # #                 factor_short_name_new, 
    # #                 factor_definition_new, 
    # #                 factor_domain_new, 
    # #                 factor_importance_new, 
    # #                 factor_controllability_new, 
    # #                 factor_strategic_new, 
    # #                 factor_uncertainty_new,
    # #                 factor_vulnerable_new,
    # #                 factor_resistant_new,
    # #                 factor_info_new
    # #                )
    # #             save_backup()



## DELETE ##

st.sidebar.header("delete relationship")

with st.sidebar.expander("delete relationship").form("delete_relationship"):
    
    relationship_name_to_delete = st.selectbox('Which relationship would you like to delete?', st.session_state.df_relationships.relationship_id)
    
    if relationship_name_to_delete != None:
        index_to_delete = st.session_state.df_relationships[st.session_state.df_relationships['relationship_id'] == relationship_name_to_delete].index.values[0]
    
    press_delete = st.form_submit_button("delete")
    
    if press_delete and relationship_name_to_delete != None:
        delete_relationship(index_to_delete)
        save_backup()
        st.experimental_rerun()

        
## SAVE ##

st.sidebar.header("save relationships")

csv = convert_df(st.session_state.df_relationships)

st.sidebar.download_button(
     label="Download relationships table as CSV",
     data=csv,
     file_name='relationships.csv',
     mime='text/csv',
 )

with st.sidebar:
    st.markdown("""---""")
    
## SAVE MAP AS HTML ##

st.sidebar.header("EXPORT MAP TO HTML")


################################################################################################################################
######################################################## MAIN PAGE #############################################################
################################################################################################################################

tab1, tab2, tab3, tab4 = st.tabs(["System Map", "Domains", "Factors", "Relationships"])

with tab1:
    st.markdown('### System Map')
    plot_relationships()

with tab2:
    st.markdown('### Domains Table')
    AgGrid(st.session_state.df_domains, fit_columns_on_grid_load=False, width='100%')
    plot_domains()

with tab3:
    st.markdown('### Factors Table')
    AgGrid(st.session_state.df_factors, fit_columns_on_grid_load=False, width='100%')
    plot_factors()

with tab4:
    st.markdown('### Relationships Table')
    AgGrid(st.session_state.df_relationships, fit_columns_on_grid_load=True, width='100%')
    plot_relationships()

    


