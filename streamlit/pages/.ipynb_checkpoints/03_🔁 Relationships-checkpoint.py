from utilities import *

st.set_page_config(page_title='Relationships', layout="wide", page_icon="🔠")

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

st.sidebar.header("Load relationships")

with st.sidebar.expander("Load relationships from Google Sheets").form("load_relationships_gsheet"):
    
    sheet_id = st.text_input('Google Sheets ID', '1YDsfTegWgBnH4KB4FJ5g7fegaaKFFItTkuOF3Fbc3Vc')
    st.warning('Warning:')
    
    press_load = st.form_submit_button("Load")
    
    if press_load:
        load_relationships(sheet_id)
        #add_factors_from_relationships_table()

with st.sidebar.expander("Load relationships from CSV file"):
    uploaded_file = st.file_uploader("Choose a relationships file")
    if uploaded_file is not None:
        df_relationships=pd.read_csv(uploaded_file)
        df_relationships = df_relationships.loc[:, ~df_relationships.columns.str.contains('^Unnamed')]
        st.session_state.df_relationships = df_relationships
    
# st.sidebar.header("Add relationships")

# with st.sidebar.expander("Add relationships").form("add_relationships"):
    
#     relationship_from =  st.selectbox()
#     relationship_to =  st.selectbox()
#     relationship_polarity =  st.radio('What is the direction of is this relationship?', ('Positive', 'Negative', 'Unclear'), horizontal = True)
#     relationship_strength =  st.radio('What is the strength of this relationship?', ('Low', 'Medium', 'High', 'Unclear'), horizontal = True)
#     relationship_temporality =  st.radio('What is the temporality of this relationship?', ('Immediate', 'Slight Delay', 'Significant Delay', 'Unclear'), horizontal = True)
#     relationship_importance = st.radio('How important is this relationship?', ('Low', 'Medium', 'High', 'Unclear'), horizontal = True)
        
#     press_add = st.form_submit_button("Add")
    
#     if press_add:
#         add_relationship(relationship_from,
#                          relationship_to,
#                          relationship_polarity, 
#                          relationship_strength, 
#                          relationship_temporality,
#                          relationship_importance)
        
# st.sidebar.header("Edit relationship")

# with st.sidebar.expander("Edit relationship"):
    
#     relationship_id_old = st.selectbox('Which relationship would you like to edit?', st.session_state.df_relationshipss.relationship_id)

#     with st.sidebar.expander("Edit relationship properties").form("edit_relationship"):

#         if relationship_id_old == None:
#             st.warning('there are no relationships to edit')

#         if relationship_id_old != None:
#             index_to_edit = st.session_state.df_relationships[st.session_state.df_relationships['relationship_id'] == relationship_id_old].index.values[0]

#             #preselected values
            
#             relationship_from_preselect = 0
#             relationship_to_preselect = 0
            

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

#             relationship_info_preselect = str(st.session_state.df_relationships.at[index_to_edit,'other_info'])   

#             factor_long_name_new =  st.text_input('Long name', value=factor_long_name_preselect)
#             factor_short_name_new =  st.text_input('Short name', value=factor_short_name_preselect)
#             factor_definition_new = st.text_input('How would you define this factor?', value=factor_definition_preselect)
#             factor_domain_new = st.selectbox('To what domain does this factor belong?', st.session_state.df_domains.name, index=factor_domain_preselect)
#             factor_importance_new = st.radio('How important is this factor?', ('Low', 'Medium', 'High', 'Unsure'), horizontal = True, index=factor_importance_preselect)
#             factor_controllability_new = st.radio('How controllable is this factor?', ('Low', 'Medium', 'High', 'Unsure'), horizontal = True, index=factor_controllability_preselect)
#             factor_strategic_new = st.radio('How strategic is this factor?', ('Low', 'Medium', 'High', 'Unsure'), horizontal = True, index=factor_strategic_preselect)
#             factor_uncertainty_new = st.radio('How uncertain is this factor?', ('Low', 'Medium', 'High', 'Unsure'), horizontal = True, index=factor_uncertainty_preselect)
#             factor_vulnerable_new = st.radio('How vulnerable to change is this factor?', ('Low', 'Medium', 'High', 'Unsure'), horizontal = True, index=factor_vulnerable_preselect)
#             factor_resistant_new = st.radio('How resistant to change is this factor?', ('Low', 'Medium', 'High', 'Unsure'), horizontal = True, index=factor_resistant_preselect)
#             factor_info_new = st.text_input('Additional info', value=factor_info_preselect)

#         press_edit = st.form_submit_button("Edit")

#         if press_edit and factor_name_old != None:
#             edit_factor(index_to_edit,
#                 factor_long_name_new, 
#                 factor_short_name_new, 
#                 factor_definition_new, 
#                 factor_domain_new, 
#                 factor_importance_new, 
#                 factor_controllability_new, 
#                 factor_strategic_new, 
#                 factor_uncertainty_new,
#                 factor_vulnerable_new,
#                 factor_resistant_new,
#                 factor_info_new
#                )
    
# st.sidebar.header("Delete relationship")
# with st.sidebar.expander("Delete relationship").form("delete_relationship"):
#     relationship_name_to_delete = st.selectbox('Which relationship would you like to delete?', st.session_state.df_relationships.long_name)
#     if relationship_id_to_delete != None:
#         index_to_delete = st.session_state.df_relationships[st.session_state.df_relationships['long_name'] == relationship_id_to_delete].index.values[0]
    
#     press_delete = st.form_submit_button("Delete")
    
#     if press_delete and relationship_name_to_delete != None:
#         delete_relationship(index_to_delete)
#         st.experimental_rerun()
        
# st.sidebar.header("Delete all")
# with st.sidebar.expander("Delete all relationships").form("delete_all_factros"):
    
#     option = st.selectbox('Are you sure?', ('Yes', 'No'))
#     if st.form_submit_button("Submit"):
#         if option == 'Yes':
#             sheet_id = '1YDsfTegWgBnH4KB4FJ5g7fegaaKFFItTkuOF3Fbc3Vc'
#             sheet_name = 'RELATIONSHIPS'
#             url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
#             df_relationships=pd.read_csv(url)
#             df_relationships = df_relationships.loc[:, ~df_relationships.columns.str.contains('^Unnamed')]
#             st.session_state.df_relationships = df_relationships
#             st.experimental_rerun()
#         if option == 'No':
#             st.experimental_rerun()

# st.sidebar.header("Save all")

# csv = convert_df(st.session_state.df_relationships)

# st.sidebar.download_button(
#      label="Download relationships table as CSV",
#      data=csv,
#      file_name='relationships.csv',
#      mime='text/csv',
#  )


################################################################################################################################
######################################################## MAIN PAGE #############################################################
################################################################################################################################

st.markdown('### Relationships Table')

placeholder_relationships_table = st.empty()

with placeholder_relationships_table.expander("", expanded=True):
    
    st.session_state.df_relationships['relationship_id'] = st.session_state.df_relationships.reset_index().index
    
    AgGrid(
        st.session_state.df_relationships)

st.markdown('### Relationships Plot')

placeholder_relationships_plot = st.empty()

with placeholder_relationships_plot.expander("", expanded=True):        
    plot_relationships()
    
save_graph(G)