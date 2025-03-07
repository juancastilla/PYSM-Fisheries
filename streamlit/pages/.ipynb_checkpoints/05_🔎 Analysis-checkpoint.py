from utilities import *

st.sidebar.header("Choose analysis")

st.sidebar.markdown("#### Node Importance")
analysis_choice_1 = st.sidebar.checkbox('Centrality Metrics')
analysis_choice_9 = st.sidebar.checkbox('Centrality Archetypes')

st.sidebar.markdown("#### Network Controllability")
analysis_choice_2 = st.sidebar.checkbox('Control Centrality: Individual Factors')
analysis_choice_3 = st.sidebar.checkbox('Control Centrality: Multiple Factors')
analysis_choice_4 = st.sidebar.checkbox('Robust Controllability (Liu et al)')
analysis_choice_5 = st.sidebar.checkbox('Global Controllability (Jia et al)')
#analysis_choice_6 = st.sidebar.checkbox('Structural permeability (Lo ludice et al)')

st.sidebar.markdown("#### Structural Analysis")
analysis_choice_7 = st.sidebar.checkbox('Intended & Unintended Consequences')
analysis_choice_8 = st.sidebar.checkbox('Archetype detection')

st.sidebar.markdown("#### Fuzzy Cognitive Maps")

with st.expander("Relationships Plot"):

    st.markdown('### What is the system?')
    
    G = plot_relationships()

if analysis_choice_1:
    
    node_colors, centrality_summary_df = load_centrality(G)
    
    with st.expander('Centrality Tables'):
        
        st.markdown('## Node Importance: Centrality')
        centrality_md = read_markdown_file("centrality_explained.md")
        st.markdown(centrality_md, unsafe_allow_html=True)
        
        st.markdown('### Centrality — Summary')

        AgGrid(centrality_summary_df.round(2))
        
        st.markdown('*white = more important | dark blue = less important*')
        HtmlFile_centrality_summary_df_styled = open('centrality_summary_df_styled.html','r',encoding='utf-8')
        components.html(HtmlFile_centrality_summary_df_styled.read(),height=2000)
        
        st.markdown('### Centrality — All Rankings')
        st.markdown('*white = more important | dark blue = less important*')
        HtmlFile_centrality_ranks_df_styled = open('centrality_ranks_df_styled.html','r',encoding='utf-8')
        components.html(HtmlFile_centrality_ranks_df_styled.read(),height=2000)
        
        st.markdown('### Centrality — Average Ranking')
        st.markdown('*white = more important | dark blue = less important*')
        HtmlFile_average_ranks_df_styled = open('average_ranks_df_styled.html','r',encoding='utf-8')
        components.html(HtmlFile_average_ranks_df_styled.read(),height=2000)
    
    f = draw_centralities(G,node_colors)
        
    with st.expander('Centrality Plots'):
        
        st.pyplot(f)
        
if analysis_choice_9:
    
    with st.expander('Centrality Archetypes'):
        
        st.markdown('## Centrality Archetypes')
        
        plot_centrality_archetypes(G)
               
        
if analysis_choice_2:
    
    with st.expander('Control Centrality: Individual Factors'):
        
        st.markdown('### To what extent can a single factor control the system?')
        
        st.markdown('##### The values for control centrality indicate the % of the system that can be potentially controlled by each factor')
    
        single_factor_control_centralities_df  = control_centrality_single(G)
        AgGrid(single_factor_control_centralities_df)

if analysis_choice_3:
    
    with st.expander('Control Centrality: Multiple Factors'):

        st.markdown('### To what extent can a policy (influencing a specific group of factors) control the system?')
    
        st.markdown('##### Use the checkboxes below to select those factors that are part of a candidate policy. The gauge will display the % of the system that can be potentially controlled by these factors')
        gd = GridOptionsBuilder.from_dataframe(st.session_state.df_factors)
        gd.configure_selection(selection_mode='multiple', use_checkbox=True)
        gridoptions = gd.build()
        grid_table = AgGrid(st.session_state.df_factors, 
                            gridOptions=gridoptions, 
                            update_mode = GridUpdateMode.SELECTION_CHANGED)
        sel_rows = grid_table["selected_rows"]
        
        if sel_rows:
            sel_rows_df = pd.DataFrame(sel_rows)
            factors = sel_rows_df.factor_id.to_list()
            multiple_factor_controllability = controllability_multiple(G,factors)
            plot_controllability_gauge(multiple_factor_controllability)

if analysis_choice_4:

    with st.expander('Robust Controllability (Liu et al)'):
        
        st.markdown('### How important is the removal of a factor in mantaining control of the whole system?')
        st.markdown('##### Indispensable, Dispensable and Neutral factors')
        st.markdown('**Overview**')
        st.markdown('In robust controllability (by Liu et al. [38], pictured in Fig. 1b), the MIS is re-calculated (size ND′) after removing each node from the network. The node is then classified by its effect on the manipulation required to control the network, where an increase in the size of the MIS makes it more difficult to control the network and a decrease in the size of the MIS makes it easier to control the network. The removal of: an indispensable node increases the number of driver nodes (ND′ > ND), a dispensable node decreases the number of driver nodes (ND′ < ND), and a neutral node has no effect on the number of driver nodes (ND′ = ND). This method has previously been applied to many network types such as gene regulatory networks, food webs, citation networks, and PPI networks to better understand what drives the dynamics of each system [29, 38]. While it is useful to observe the structural changes to the network after the removal of singular nodes, this method only considers one possible MIS.')
        
        G_bipartite, left_nodes = directed_to_bipartite(G)
        ND, unmatched_factors = N_D(G, G_bipartite, left_nodes, 1)
        liu_class_summary_df = compute_liu_classes(G, G_bipartite,ND)
        st.dataframe(liu_class_summary_df)
        #AgGrid(liu_class_summary_df)
    
if analysis_choice_5:

    with st.expander('Global Controllability (Jia et al)'):

        st.markdown('### How important is a factor in controlling the whole system?')
        st.markdown('##### Critical, Intermittent and Redundant factors')        
        st.markdown('**Overview**')
        st.markdown('A second global controllability method by Jia et al. [39] (Pictured in Fig. 1c) classifies a node by its role across all possible MISs. A critical node is included in all possible MISs, an intermittent node is included in some possible MISs, and a redundant node is not included in any possible MISs. This method places each node in the broader context of all possible control configurations.')
             
        G_bipartite, left_nodes = directed_to_bipartite(G)
        ND, unmatched_factors = N_D(G, G_bipartite, left_nodes, 1)
        jia_class_summary_df = compute_jia_classes(G,G_bipartite,ND)
        N = G.number_of_nodes()
        
        col1, col2, col3 = st.columns([3, 6, 0.1])
        
        message = '##### To fully control this system we need a minimum of: ' + str(ND) + ' factors' + ' (from a total of ' + str(N) + ' factors)'
        
        col1.markdown(message)
        
        col1.dataframe(jia_class_summary_df.style.apply(highlight_jia_classes, axis=1),width=400)
        #AgGrid(jia_class_summary_df)
        

        col2.markdown('##### Minimum Input/Driver Sets')
        col2.markdown('')

        MIS_df = compute_all_MIS(G,jia_class_summary_df,ND)
        with col2:
            AgGrid(MIS_df,height=1000)
        
if analysis_choice_7:

    with st.expander('Path analysis'):
        
        st.markdown('### What are the intended and unintended consequences of a given intervention or policy?')
        
        source_factor_name = st.selectbox('Which factor does this policy directly affect?', st.session_state.df_factors.long_name, index=1)
        target_factor_name = st.selectbox('Which factor is this policy aiming to control?', st.session_state.df_factors.long_name, index=0)
        
        path_intended = []
        path_unintended = []
        
        reload_data = False
        if st.button('Load paths'):
            reload_data = True
        
        source_index = st.session_state.df_factors[st.session_state.df_factors['long_name'] == source_factor_name].index.values[0]
        target_index = st.session_state.df_factors[st.session_state.df_factors['long_name'] == target_factor_name].index.values[0]

        intended_df, unintended_df = compute_icucpath_analysis(G, source_index, target_index)
         
        st.markdown('##### Use the checkboxes below to select those factors that are part of a candidate policy')

        gd_intended = GridOptionsBuilder.from_dataframe(intended_df)
        gd_intended.configure_selection(selection_mode='single', use_checkbox=True)
        gridoptions_intended = gd_intended.build()
        
        gd_unintended = GridOptionsBuilder.from_dataframe(unintended_df)
        gd_unintended.configure_selection(selection_mode='single', use_checkbox=True)
        gridoptions_unintended = gd_unintended.build()

        col1, col2 = st.columns(2)

        with col1:

            grid_table_intended = AgGrid(intended_df, gridOptions=gridoptions_intended, key='intended_df', reload_data=reload_data, update_mode = 'MANUAL', height=500)
            
            sel_row_intended = grid_table_intended["selected_rows"]
                    
            if sel_row_intended:
                sel_row_intended_df = pd.DataFrame(sel_row_intended)
                path_intended = sel_row_intended_df.Path.at[0]
                
                if platform.system() == 'Darwin':
                    path_intended = ast.literal_eval(path_intended)
                else:
                    path_intended = eval(str(path_intended))
                
                path_to_sentence(G,path_intended)
            
        with col2:
            
            grid_table_unintended = AgGrid(unintended_df, gridOptions=gridoptions_unintended, key='unintended_df', reload_data=reload_data, update_mode = 'MANUAL', height=500)
            
            sel_row_unintended = grid_table_unintended["selected_rows"]
                    
            if sel_row_unintended:
                sel_row_unintended_df = pd.DataFrame(sel_row_unintended)
                path_unintended = sel_row_unintended_df.Path.at[0]
                
                if platform.system() == 'Darwin':
                    path_unintended = ast.literal_eval(path_unintended)
                else:
                    path_unintended = eval(str(path_unintended))
                
                path_to_sentence(G,path_unintended)
        
        if sel_row_intended or sel_row_unintended: plot_icucpaths(G,path_intended,path_unintended)

            
if analysis_choice_8:            
            
    with st.expander('Problem Archetype Detection'):
        
        st.markdown('### Does this system have generic structural problems and how can they be addressed?')
        
        key_factor_name = st.selectbox('Choose a key variable of interest?', st.session_state.df_factors.long_name, index=0)
        archetype_name = st.selectbox('Choose a generic problem archetype', ['Underachievement', 'Relative Achievement', 'Relative Control', 'Out of Control'], index=0)
        
        target_index = st.session_state.df_factors[st.session_state.df_factors['long_name'] == key_factor_name].index.values[0]
                
        reload_data = False   
        if st.button('Load analysis'):
            reload_data = True
            
        st.markdown('##### Analysis for ' + archetype_name + ' archetype: ' + key_factor_name)

        if archetype_name == 'Underachievement':
            bool_ic = True
            bool_uc = False

        if archetype_name == 'Relative Achievement':
            bool_ic = True
            bool_uc = True

        if archetype_name == 'Relative Control':
            bool_ic = False
            bool_uc = False

        if archetype_name == 'Out of Control':
            pass

        archetype_loops_df = find_ADAS_archetypes(G, target_index, bool_ic, bool_uc) 
        archetype_df = compute_archetype_dataframe(G,archetype_loops_df)

        gd_archetypes = GridOptionsBuilder.from_dataframe(archetype_df)
        gd_archetypes.configure_selection(selection_mode='single', use_checkbox=True)
        gridoptions_archetypes = gd_archetypes.build()

        grid_table_archetypes = AgGrid(archetype_df, gridOptions=gridoptions_archetypes, key='archetypes_df', reload_data=reload_data, update_mode = 'MANUAL')
        reload_data = False

        sel_row_archetype = grid_table_archetypes["selected_rows"]

        ic_path = []
        uc_path = []

        if sel_row_archetype:
            
            sel_row_archetype_df = pd.DataFrame(sel_row_archetype)
            ic_path = sel_row_archetype_df.ic_Path.at[0]
            uc_path = sel_row_archetype_df.uc_Path.at[0]
            
            if platform.system() == 'Darwin':
                ic_path = ast.literal_eval(ic_path)
                uc_path = ast.literal_eval(uc_path)
            else:
                ic_path = eval(str(ic_path))
                uc_path = eval(str(uc_path))

        col1, col2 = st.columns(2)

        with col1: 
            
                if sel_row_archetype:
                    
                    path_to_sentence(G,ic_path)
                    path_polarity = compute_path_polarity(G, ic_path)
                    st.write('')
                    if path_polarity == 1: st.write('➡️ This is a **REINFORCING** loop')
                    if path_polarity == -1: st.write('➡️ This is a **BALANCING** loop')
            
        with col2: 
            
            if sel_row_archetype:
                
                path_to_sentence(G,uc_path)
                path_polarity = compute_path_polarity(G, uc_path)
                st.write('')
                if path_polarity == 1: st.write('➡️ This is a **REINFORCING** loop')
                if path_polarity == -1: st.write('➡️ This is a **BALANCING** loop')
                
        if sel_row_archetype: plot_icucpaths(G,ic_path,uc_path)