from utilities import *

load_factors(st.session_state.sheet_id)
load_relationships(st.session_state.sheet_id)
load_domains(st.session_state.sheet_id)

st.sidebar.header("Choose analysis")

st.sidebar.markdown("#### Map visualisation")
analysis_choice_0 = st.sidebar.checkbox('CLD', key='0')
analysis_choice_1 = st.sidebar.checkbox('Force-directed graph', key='1')
analysis_choice_2 = st.sidebar.checkbox('Chord diagram', key='2')

st.sidebar.markdown("#### Structural Analysis")
analysis_choice_3 = st.sidebar.checkbox('Upstream/Downstream Submaps', key='3')

st.sidebar.markdown("#### Hierarchical clustering")
analysis_choice_4 = st.sidebar.checkbox('Dendograms', key='4')

st.sidebar.markdown("#### Node Importance")
analysis_choice_5 = st.sidebar.checkbox('Centrality Metrics', key='5')
analysis_choice_16 = st.sidebar.checkbox('Centrality Clustermaps', key='16')
analysis_choice_6 = st.sidebar.checkbox('Centrality Archetypes', key='6')

st.sidebar.markdown("#### Network Controllability")
analysis_choice_7 = st.sidebar.checkbox('Control Centrality', key='7')
analysis_choice_9 = st.sidebar.checkbox('Robust Controllability (Liu method)', key='9')
analysis_choice_10 = st.sidebar.checkbox('Global Controllability (Jia method)', key='10')
#analysis_choice_6 = st.sidebar.checkbox('Structural permeability (Lo ludice et al)')

st.sidebar.markdown("#### Path Analysis")
analysis_choice_11 = st.sidebar.checkbox('Intended & Unintended Consequences', key='11')


st.sidebar.markdown("#### Tradeoff Analysis")
analysis_choice_13 = st.sidebar.checkbox('Interactive Tradeoff Plot', key='13')

st.sidebar.divider()

st.sidebar.header("To be implemented")

st.sidebar.markdown("#### Archetype detection")
analysis_choice_12 = st.sidebar.checkbox('Archetype detection', key='12')

st.sidebar.markdown("#### Scenario Analysis")
analysis_choice_14 = st.sidebar.checkbox('Fuzzy Cognitive Mapping', key='14')

st.sidebar.markdown("#### FishGPT")
analysis_choice_15 = st.sidebar.checkbox('FishGPT QA Chatbot', key='15')

st.title(st.session_state.fishery)

if analysis_choice_0:

    with st.expander('CLD Diagram'):

        if platform.system() == 'Darwin':

            if st.session_state.fishery == 'Octopus Chile':
                filename = 'OctopusChile.png'
            if st.session_state.fishery == 'Octopus Peru':
                filename = 'OctopusPeru.png'
            if st.session_state.fishery == 'Southern Hake':
                filename = 'SouthernHake.png'
            if st.session_state.fishery == 'Jumbo Flying Squid':
                filename = 'JumboFlyingSquid.png'
            if st.session_state.fishery == 'Anchoveta':
                filename = 'Anchoveta.png'

        else:

            if st.session_state.fishery == 'Octopus Chile':
                filename = 'streamlit/OctopusChile.png'
            if st.session_state.fishery == 'Octopus Peru':
                filename = 'streamlit/OctopusPeru.png'
            if st.session_state.fishery == 'Southern Hake':
                filename = 'streamlit/SouthernHake.png'
            if st.session_state.fishery == 'Jumbo Flying Squid':
                filename = 'streamlit/JumboFlyingSquid.png'
            if st.session_state.fishery == 'Anchoveta':
                filename = 'streamlit/Anchoveta.png'

        st.title('Causal Loop Diagram')
        st.image(filename, use_column_width=True)

if analysis_choice_1:

    with st.expander('Force-directed graph'):

        FORCEDIRECTED_rel_choice = st.selectbox('Choose which relationships to display', ('All relationships', 'Strong only'), index=0, key='FORCEDIRECTED_rel_choice')
        FORCEDIRECTED_isolates_choice = st.checkbox('Hide isolate nodes', key='FORCEDIRECTED_isolates_choice')

        tab1, tab2, tab3, tab4 = st.tabs(["System Map", "Domains", "Factors", "Relationships"])

        with tab1:
            st.markdown('### System Map')
            G=plot_relationships(FORCEDIRECTED_rel_choice,FORCEDIRECTED_isolates_choice,'display')

        with tab2:
            st.markdown('### Domains Table')
            AgGrid(st.session_state.df_domains, fit_columns_on_grid_load=False, width='100%')
            # plot_domains()

        with tab3:
            st.markdown('### Factors Table')
            AgGrid(st.session_state.df_factors, fit_columns_on_grid_load=False, width='100%')
            # plot_factors()

        with tab4:
            st.markdown('### Relationships Table')
            AgGrid(st.session_state.df_relationships, fit_columns_on_grid_load=True, width='100%')
            G=plot_relationships(FORCEDIRECTED_rel_choice,FORCEDIRECTED_isolates_choice,'display')

if analysis_choice_2:

    with st.expander('Chord diagram'):

        G=plot_relationships("Strong Only",True,'no_display')

        from plotapi import Chord

        Chord.api_key("573968cb-86f2-4a43-991d-aa2b5d6974a4")
        matrix = nx.to_numpy_matrix(G, weight='edge_value').tolist()
        names = list(nx.get_node_attributes(G,"label").values())
        colors = list(nx.get_node_attributes(G,"color").values())
        

            # Save and read graph as HTML file (on Streamlit Sharing)
        try:
            Chord(matrix, names, directed=True, colors=colors, reverse_gradients=True, popup_names_only=False, font_size="6px", width=1500, margin=300, rotate=75, label_colors='black').to_html('./streamlit/chord_graph.html')
            HtmlFile = open('./streamlit/chord_graph.html','r',encoding='utf-8')
            
            # Save and read graph as HTML file (locally)
        except:
            path = 'html_files'
            Chord(matrix, names, directed=True, colors=colors, reverse_gradients=True, popup_names_only=False, font_size="6px", width=1500, margin=300, rotate=75, label_colors='black').to_html(f'{path}/chord_graph.html')
            HtmlFile = open(f'{path}/chord_graph.html','r',encoding='utf-8')

                # nt.show('G_factors_and_relationships.html')
                # HtmlFile = open('G_factors_and_relationships.html','r',encoding='utf-8')
        components.html(HtmlFile.read(),height=1800)
                # save_graph(G)


if analysis_choice_3:
        
    with st.expander('Submaps'):

        col1, col2, col3 = st.columns(3)

        with col1:
            SUBMAP_rel_choice = st.selectbox('Choose which relationships to display', ('All relationships', 'Strong only'), index=0, key='SUBMAP_rel_choice')

        with col2:
            SUBMAP_steps_choice = st.selectbox('How many steps upstream/downstream?', (1, 2, 3), index=0, key='SUBMAP_steps_choice')

        with col3:
            SUBMAP_factor_choice = st.selectbox('Choose a factor:', st.session_state.df_factors.long_name, index=0, key='SUBMAP_factor_choice')
            selected_factor_id = st.session_state.df_factors[st.session_state.df_factors['long_name'] == SUBMAP_factor_choice]['factor_id'].values[0]

        G=plot_relationships_submaps(SUBMAP_rel_choice, SUBMAP_steps_choice, selected_factor_id)

if analysis_choice_4:
        
    with st.expander('Hierachical clustering'):

        DENDOGRAM_rel_choice = st.selectbox('Choose which relationships to display', ('All relationships', 'Strong only'), index=0, key='SUBMAP_rel_choice')

        if DENDOGRAM_rel_choice == 'All relationships':

            st.header('Dendogram')

            G=plot_relationships('All relationships',True,'no_display')

            adjacency_matrix = nx.adjacency_matrix(G)
            adjacency = adjacency_matrix

            # hierarchical clustering — Paris
            paris = Paris()
            dendrogram = paris.fit_predict(adjacency)

            svg = svg_dendrogram(dendrogram, names=list(nx.get_node_attributes(G,"label").values()), rotate=True, width=700, height=1400, n_clusters=5, font_size=20)

            render_svg(svg)

        if DENDOGRAM_rel_choice == 'Strong only':

            st.header('Dendogram')

            G=plot_relationships('Strong only',True,'no_display')

            G.remove_nodes_from(list(nx.isolates(G)))
            largest = max(nx.weakly_connected_components(G), key=len)
            G = G.subgraph(largest)

            adjacency_matrix = nx.adjacency_matrix(G)
            adjacency = adjacency_matrix

            # hierarchical clustering — Paris
            paris = Paris()
            dendrogram = paris.fit_predict(adjacency)

            svg = svg_dendrogram(dendrogram, names=list(nx.get_node_attributes(G,"label").values()), rotate=True, width=700, height=1500, n_clusters=5, font_size=20)

            render_svg(svg)

if analysis_choice_5:
    
    G=plot_relationships('All relationships',False,'no_display')

    centrality_summary_df_styled, centrality_ranks_df_styled, average_ranks_df_styled, centrality_summary_df, centrality_ranks_df = load_centrality(G)

    # node_colors, centrality_summary_df = load_centrality(G)
    
    with st.expander('Centrality Tables'):

        col1,col2 =  st.columns(2)
        
        # st.markdown('## Node Importance: Centrality')
        # centrality_md = read_markdown_file("centrality_explained.md")
        # st.markdown(centrality_md, unsafe_allow_html=True)
        
        with col1:
            st.markdown('### Centrality — Summary')
            st.markdown('*white = more important | dark blue = less important*')
            st.dataframe(centrality_summary_df_styled,width=400, use_container_width=True)

        with col2:
            st.markdown('### Centrality — All Rankings')
            st.markdown('*white = more important | dark blue = less important*')    
            st.dataframe(centrality_ranks_df_styled,width=400, use_container_width=True)
        
        st.markdown('### Centrality — Average Ranking')
        st.markdown('*white = more important | dark blue = less important*')
        st.dataframe(average_ranks_df_styled,width=400, use_container_width=True)
    
    f = draw_centralities(G)
        
    with st.expander('Centrality Plots'):

        col1, col2 = st.columns([0.7, 0.3], gap="small")

        with col1:
        
            st.pyplot(f)

        with col2:

            df =  centrality_summary_df.copy()
            df = df['label']
            st.dataframe(df, use_container_width=True)
        
    with st.expander('Centrality Correlations'):

        col1, col2 = st.columns(2, gap="small")

        sns.set(font="Arial")
        sns.set_style("ticks")  

        with col1:
            st.title("Pairplot")
            pairplot_dict = {0:"yellow", 1:"cornflowerblue", 2: 'magenta', 3: 'lightcoral', 4: 'orange', 5: 'purple'}
            pairplot = sns.pairplot(centrality_summary_df, hue='domain', palette=pairplot_dict,  corner=True)
            st.pyplot(pairplot)
        
        with col2:
            st.title("Heatmap")
            corr = centrality_summary_df.drop('domain', axis=1).corr()
            fig, ax = plt.subplots() #solved by add this line 
            ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap="Blues", annot=True)
            st.pyplot(fig)

if analysis_choice_6:
    
    with st.expander('Centrality Archetypes'):
        
        st.markdown('## Centrality Archetypes')
        
        G=plot_relationships('All relationships',False,'no_display')

        plot_centrality_archetypes(G)
               
if analysis_choice_7:
    
    CONTROLCENTRALITY_rel_choice = st.selectbox('Choose which relationships to display', ('All relationships', 'Strong only'), index=0, key='CONTROLCENTRALITY_rel_choice')
    CONTROLCENTRALITY_isolates_choice = st.checkbox('Hide isolate nodes', key='CONTROLCENTRALITY_isolates_choice')

    with st.expander('Control Centrality: Individual Factors'):
        
        st.markdown('### To what extent can a single factor control the system?')

        G=plot_relationships(CONTROLCENTRALITY_rel_choice,CONTROLCENTRALITY_isolates_choice,'no_display')
        
        st.markdown('##### The values for control centrality indicate the % of the system that can be potentially controlled by each factor')
        
        largest = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest)
        single_factor_control_centralities_df  = control_centrality_single(G)

        # Sort the dataframe by the 'control_centrality' column in ascending order
        sorted_df = single_factor_control_centralities_df.sort_values('control_centrality', ascending=False)

        #st.dataframe(sorted_df)

        # Plot the horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 15))  # Increase the figure size
        sorted_df.plot.barh(y='control_centrality', x='label', ax=ax, legend=False)
        ax.set_title('Control Centrality')
        ax.set_xlabel('Control Centrality Value')
        ax.set_ylabel('Factors')

        st.pyplot(fig)  # Display the plot in Streamlit
    
    with st.expander('Control Centrality: Multiple Factors'):

        st.markdown('### To what extent can a policy (influencing a specific group of factors) control the system?')
    
        col1, col2 = st.columns(2)

        with col1:

            largest = max(nx.weakly_connected_components(G), key=len)
            G = G.subgraph(largest)

            factors = list(G.nodes())
            f_list = pd.DataFrame(factors,columns=['factor_id']).factor_id.to_list()
            all_factors = st.session_state.df_factors.drop(['domain_id','domain_name','short_name','Comentario'], axis=1)
            subfactors_df = all_factors[all_factors['factor_id'].isin(f_list)]

            st.markdown('##### Use the checkboxes below to select those factors that are part of a candidate policy. The gauge will display the % of the system that can be potentially controlled by these factors')
            gd = GridOptionsBuilder.from_dataframe(subfactors_df)
            gd.configure_selection(selection_mode='multiple', use_checkbox=True)
            gridoptions = gd.build()
            grid_table = AgGrid(subfactors_df, 
                                gridOptions=gridoptions, 
                                update_mode = GridUpdateMode.SELECTION_CHANGED)
            sel_rows = grid_table["selected_rows"]
        
        with col2:

            # G=plot_relationships(CONTROLCENTRALITY_rel_choice,True,'no_display')
            # largest = max(nx.weakly_connected_components(G), key=len)
            # G = G.subgraph(largest)

            if sel_rows:
                sel_rows_df = pd.DataFrame(sel_rows)
                factors = sel_rows_df.factor_id.to_list()
                multiple_factor_controllability = controllability_multiple(G,subfactors_df)
                plot_controllability_gauge(multiple_factor_controllability)

if analysis_choice_9:

    with st.expander('Robust Controllability (Liu et al)'):
        
        st.markdown('### How important is the removal of a factor in mantaining control of the whole system?')
        st.markdown('##### Indispensable, Dispensable and Neutral factors')
        # st.markdown('**Overview**')
        # st.markdown('In robust controllability (by Liu et al. [38], pictured in Fig. 1b), the MIS is re-calculated (size ND′) after removing each node from the network. The node is then classified by its effect on the manipulation required to control the network, where an increase in the size of the MIS makes it more difficult to control the network and a decrease in the size of the MIS makes it easier to control the network. The removal of: an indispensable node increases the number of driver nodes (ND′ > ND), a dispensable node decreases the number of driver nodes (ND′ < ND), and a neutral node has no effect on the number of driver nodes (ND′ = ND). This method has previously been applied to many network types such as gene regulatory networks, food webs, citation networks, and PPI networks to better understand what drives the dynamics of each system [29, 38]. While it is useful to observe the structural changes to the network after the removal of singular nodes, this method only considers one possible MIS.')
        
        col1, col2 = st.columns(2)

        with col1:

            G=plot_relationships('Strong only',True,'display')
            largest = max(nx.weakly_connected_components(G), key=len)
            G = G.subgraph(largest)

        with col2:
            G_bipartite, left_nodes = directed_to_bipartite(G)
            ND, unmatched_factors = N_D(G, G_bipartite, left_nodes, 1)
            liu_class_summary_df = compute_liu_classes(G, G_bipartite,ND)
            st.dataframe(liu_class_summary_df.style.apply(highlight_liu_classes, axis=1), use_container_width=True)
            #AgGrid(liu_class_summary_df)
    
if analysis_choice_10:

    with st.expander('Global Controllability (Jia et al)'):

        st.markdown('### How important is a factor in controlling the whole system?')
        st.markdown('##### Critical, Intermittent and Redundant factors')        
        # st.markdown('**Overview**')
        # st.markdown('A second global controllability method by Jia et al. [39] (Pictured in Fig. 1c) classifies a node by its role across all possible MISs. A critical node is included in all possible MISs, an intermittent node is included in some possible MISs, and a redundant node is not included in any possible MISs. This method places each node in the broader context of all possible control configurations.')

        G=plot_relationships('Strong only',True,'no_display')
        largest = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest)

        G_bipartite, left_nodes = directed_to_bipartite(G)
        ND, unmatched_factors = N_D(G, G_bipartite, left_nodes, 1)
        jia_class_summary_df = compute_jia_classes(G,G_bipartite,ND)
        N = G.number_of_nodes()
        
        col1, col2, col3 = st.columns([3, 6, 0.1])
        
        message = '##### To fully control this system we need a minimum of: ' + str(ND) + ' factors' + ' (from a total of ' + str(N) + ' factors)'
        
        col1.markdown(message)
        
        col1.dataframe(jia_class_summary_df.style.apply(highlight_jia_classes, axis=1),width=400)
        #AgGrid(jia_class_summary_df)
        
        display_MIS = st.checkbox('Compute and display all MISs', key='display_MIS')

        if display_MIS:
            col2.markdown('##### Minimum Input/Driver Sets')
            col2.markdown('')

            MIS_df = compute_all_MIS(G,jia_class_summary_df,ND)
            with col2:
                AgGrid(MIS_df,height=1000)
        
if analysis_choice_11:

    with st.expander('Path analysis'):
        
        st.markdown('### What are the intended and unintended consequences of a given intervention or policy?')

        G=plot_relationships('Strong only',True,'no_display')

        result_ic, result_uc, result_both, node_pairs_names  = icuc_heatmap(G)

        col1, col2, col3 = st.columns(3)

        with col1:

            plt.figure(figsize=(15,15))
            plot_ic = sns.heatmap(result_ic, annot=True, fmt="g", cmap='mako', square=True, linewidths=1, linecolor='black', clip_on=False, cbar=True, cbar_kws={"shrink": .82})
            plt.title('Same direction', size=20, weight='bold')
            st.pyplot(plot_ic.get_figure())

        with col2:

            plt.figure(figsize=(15,15))
            plot_uc = sns.heatmap(result_uc, annot=True, fmt="g", cmap='mako', square=True, linewidths=1, linecolor='black', clip_on=False, cbar=True, cbar_kws={"shrink": .82})
            plt.title('Opposite direction', size=20, weight='bold')
            st.pyplot(plot_uc.get_figure())

        with col3:

            plt.figure(figsize=(15,15))
            plot_both = sns.heatmap(result_both, annot=True, fmt="g", cmap='mako', square=True, linewidths=1, linecolor='black', clip_on=False, cbar=True, cbar_kws={"shrink": .82})
            plt.title('Conflicting intended/unintended pathways', size=20, weight='bold')
            st.pyplot(plot_both.get_figure())

        st.markdown('##### The following Control-Target node pairs with conflicting intended/unintended pathways were found:')
        # Convert list of tuples into dataframe
        df = pd.DataFrame(node_pairs_names, columns=['Control Node', 'Target Node'])

        # Display dataframe as a table in Streamlit
        st.dataframe(df)

        st.markdown('##### Which of these would you like to visualise?')

        pair_selected = st.selectbox('Choose a pair of nodes', node_pairs_names, index=0)

        source_factor_name = pair_selected[0]
        target_factor_name = pair_selected[1]
        
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
                    path_intended = eval(str(path_intended))
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
                    path_unintended = eval(str(path_unintended))
                else:
                    path_unintended = eval(str(path_unintended))
                
                path_to_sentence(G,path_unintended)
        
        if sel_row_intended or sel_row_unintended: plot_icucpaths(G,path_intended,path_unintended)
         
if analysis_choice_12:            
            
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

if analysis_choice_16:
    
        with st.expander('Centrality Clustermaps'):
            
            st.markdown('### How are the factors clustered in terms of their centrality?')

            col1, col2 = st.columns(2)
            
            with col1:
                st.title('All relationships')
                G=plot_relationships('All relationships',False,'no_display')
                centrality_summary_df_styled, centrality_ranks_df_styled, average_ranks_df_styled, centrality_summary_df, centrality_ranks_df = load_centrality(G)
                df = centrality_summary_df.drop('domain', axis=1).set_index('label')
                clustermap = sns.clustermap(df, standard_scale=1, metric="euclidean", figsize=(10,20), method='ward', robust=True, cmap='inferno')
                st.pyplot(clustermap)
    
            with col2:
                st.title('Strong only')
                G=plot_relationships('Strong only',True,'no_display')
                largest = max(nx.weakly_connected_components(G), key=len)
                G = G.subgraph(largest)
                centrality_summary_df_styled, centrality_ranks_df_styled, average_ranks_df_styled, centrality_summary_df, centrality_ranks_df = load_centrality(G)
                df = centrality_summary_df.drop('domain', axis=1).set_index('label')
                clustermap = sns.clustermap(df, standard_scale=1, metric="euclidean", figsize=(10,20), method='ward', robust=True, cmap='inferno')
                st.pyplot(clustermap)

if analysis_choice_13:

    with st.expander('Interactive Tradeoff Plot'):

        st.markdown('### What are the tradeoffs between the various factor attributes we have computed?')

        PCP_rel_choice = st.selectbox('Choose which relationships to display', ('All relationships', 'Strong only'), index=0, key='PCP_rel_choice')

        pcp_preprocess(PCP_rel_choice)