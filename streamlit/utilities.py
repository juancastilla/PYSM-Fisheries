import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pyvis import network as net
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from pathlib import Path
import os
import platform
import control
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy
import itertools
import ast
import gspread
# import googletrans
# from googletrans import Translator

### PATH CONFIGURATION ###

if platform.system() == 'Darwin':
    main_path = Path(".")
else:
    main_path = Path("./streamlit")

### MATPLOTLIB CONFIGURATION ###

matplotlib.rcParams['font.sans-serif'] = "DIN Alternate"
matplotlib.rcParams['font.family'] = "sans-serif"

### HELPER FUNCTIONS ###

def read_markdown_file(markdown_file):
    return Path(str(main_path.joinpath(markdown_file))).read_text()

def convert_df(df):
   return df.to_csv().encode('utf-8')

### AUTOSAVE BACKUP TO GOOGLE SHEETS ###


def save_backup():
    
    st.session_state.domains_backup_wks.clear()
    st.session_state.factors_backup_wks.clear()
    st.session_state.relationships_backup_wks.clear()
    
    st.session_state.df_domains = st.session_state.df_domains.fillna('')
    st.session_state.df_factors = st.session_state.df_factors.fillna('')
    st.session_state.df_relationships = st.session_state.df_relationships.fillna('')
    
    st.session_state.domains_backup_wks.update([st.session_state.df_domains.columns.values.tolist()] + st.session_state.df_domains.values.tolist())
    st.session_state.factors_backup_wks.update([st.session_state.df_factors.columns.values.tolist()] + st.session_state.df_factors.values.tolist())
    st.session_state.relationships_backup_wks.update([st.session_state.df_relationships.columns.values.tolist()] + st.session_state.df_relationships.values.tolist())

def load_backup():
    
    st.session_state.df_domains = pd.DataFrame(st.session_state.domains_backup_wks.get_all_records())
    st.session_state.df_factors = pd.DataFrame(st.session_state.factors_backup_wks.get_all_records())
    st.session_state.df_relationships = pd.DataFrame(st.session_state.relationships_backup_wks.get_all_records())
    
    st.session_state.domains_counter = len(st.session_state.df_domains.index)
    st.session_state.factors_counter = len(st.session_state.df_factors.index)
    st.session_state.relationships_counter = len(st.session_state.df_relationships.index)
    

### AUTOMATIC TRANSLATION USING GOOGLETRANS API ###

# translator = Translator()

# def translate_factors(df_en,src,dest):
    
#     df_src = df_en.copy()
#     translations = {}
#     unique_elements = df_src['long_name'].unique()
#     for element in unique_elements:
#         translations[element] = translator.translate(element, src=src, dest=dest).text
        
#     df_src.replace(translations, inplace = True)
    
#     return df_src

    
######################################################################################################
##################################### MAIN APP FUNCTIONS #############################################
######################################################################################################

#########################
####### DOMAINS #########
#########################

# Load empty domains template
def load_domains_template_table():
    
    if 'df_domains' not in st.session_state:
        sheet_id = '1YDsfTegWgBnH4KB4FJ5g7fegaaKFFItTkuOF3Fbc3Vc'
        sheet_name = 'DOMAINS'
        url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
        df_domains=pd.read_csv(url)
        df_domains = df_domains.loc[:, ~df_domains.columns.str.contains('^Unnamed')]
        st.session_state.df_domains = df_domains
        
    
# Load domains from pre-filled Google Sheets template (input is sheet id)
def load_domains(sheet_id):
    
    if st.session_state.language == 'English':
        sheet_name = 'DOMAINS'
    if st.session_state.language == 'Finnish':
        sheet_name = 'DOMAINS_fi'
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    df_domains=pd.read_csv(url)
    df_domains = df_domains.loc[:, ~df_domains.columns.str.contains('^Unnamed')]
    st.session_state.df_domains = df_domains

# Add new domain
def add_domain(name_to_add, colour_to_add, other_info_to_add):
    
    st.session_state.domains_counter = st.session_state.domains_counter + 1
    
    new_domain = {'domain_id': st.session_state.domains_counter, 'domain_name': name_to_add, 'domain_colour': colour_to_add, 'domain_info': other_info_to_add}
    st.session_state.df_domains = st.session_state.df_domains.append(new_domain, ignore_index=True)
    
# Edit existing domain
def edit_domain(index_to_edit, domain_name_new, domain_colour_new, domain_other_new):
    
    st.session_state.df_domains.at[index_to_edit,'domain_name'] = domain_name_new
    st.session_state.df_domains.at[index_to_edit,'domain_colour'] = domain_colour_new
    st.session_state.df_domains.at[index_to_edit,'domain_info'] = domain_other_new

# Delete existing domain
def delete_domain(index_to_delete):
    
    st.session_state.df_domains = st.session_state.df_domains.drop(index_to_delete)

def plot_domains():
    
    G=nx.empty_graph()

    for index, row in st.session_state.df_domains.iterrows():
        if row['domain_id'] == 0:
            size = 40
            color='yellow'
        if row['domain_id'] == 1:
            size=15
            color='cornflowerblue'
        G.add_node(row['domain_id'], label=row['domain_name'], color=color, size=size, shadow=True, font={'size': 20})
        
    nt = net.Network(width='1500px', height='1000px', directed=True)
    nt.from_nx(G)
    nt.force_atlas_2based(gravity=-300)
    #nt.show_buttons(filter_=['physics'])
    nt.show("G_domains.html")
    HtmlFile = open('G_domains.html','r',encoding='utf-8')
    components.html(HtmlFile.read(),height=1000)
    
    
#########################
####### FACTORS #########
#########################
    
    
# Load empty domains template
def load_factors_template_table():
    
    if 'df_factors' not in st.session_state:
        sheet_id = '1YDsfTegWgBnH4KB4FJ5g7fegaaKFFItTkuOF3Fbc3Vc'
        sheet_name = 'FACTORS'
        url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
        df_factors=pd.read_csv(url)
        df_factors = df_factors.loc[:, ~df_factors.columns.str.contains('^Unnamed')]
        st.session_state.df_factors = df_factors
        
    
# Load factors from pre-filled Google Sheets template (input is sheet id)
def load_factors(sheet_id):
    
    if st.session_state.language == 'English':
        sheet_name = 'FACTORS'
    if st.session_state.language == 'Finnish':
        sheet_name = 'FACTORS_fi_FINAL'
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    df_factors=pd.read_csv(url)
    df_factors = df_factors.loc[:, ~df_factors.columns.str.contains('^Unnamed')]
    st.session_state.df_factors = df_factors
    st.session_state.df_factors['factor_id'] = st.session_state.df_factors.index

# Populate domains table when factors table is loaded
# def add_domains_from_factors_table():
#     for index, row in st.session_state.df_factors.iterrows():
#         st.session_state.df_domains = st.session_state.df_domains.append({'domain_id':'', 'name':row['domain'], 'colour':'', 'other_info':''}, ignore_index=True)
#     st.session_state.df_domains['domain_id'] = st.session_state.df_domains.index


# Add new factor
def add_factor(long_name_to_add, 
               short_name_to_add, 
               definition_to_add, 
               domain_to_add, 
               importance_to_add, 
               controllability_to_add, 
               strategic_to_add, 
               uncertainty_to_add,
               vulnerable_to_add,
               resistant_to_add,
               info_to_add
              ):
    
    factor_domain_id = st.session_state.df_domains[st.session_state.df_domains['domain_name'] == domain_to_add].index.values[0]
    
    st.session_state.factors_counter = st.session_state.factors_counter + 1
    
    new_factor = {'factor_id': st.session_state.factors_counter,
                  'long_name': long_name_to_add,
                  'short_name': short_name_to_add,
                  'definition': definition_to_add,
                  'domain_name': domain_to_add,
                  'domain_id': factor_domain_id,                                  
                  'importance': importance_to_add,
                  'controllability': controllability_to_add,
                  'strategic_importance': strategic_to_add,
                  'uncertainty': uncertainty_to_add,
                  'vulnerable_to_change': vulnerable_to_add,
                  'resistant_to_change': resistant_to_add,
                  'other_info': info_to_add
                 }
    
    st.session_state.df_factors = st.session_state.df_factors.append(new_factor, ignore_index=True)

    
# Preselect factor domain when editing
def preselect_factor_domain_edit():
    pass

# Edit existing factor
def edit_factor(index_to_edit,
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
               ):
    
    factor_domain_id = st.session_state.df_domains[st.session_state.df_domains['domain_name'] == factor_domain_new].index.values[0]
    
    st.session_state.df_factors.at[index_to_edit,'long_name'] = factor_long_name_new
    st.session_state.df_factors.at[index_to_edit,'short_name'] = factor_short_name_new
    st.session_state.df_factors.at[index_to_edit,'definition'] = factor_definition_new
    st.session_state.df_factors.at[index_to_edit,'domain_name'] = factor_domain_new
    st.session_state.df_factors.at[index_to_edit,'domain_id'] = factor_domain_id                         
    st.session_state.df_factors.at[index_to_edit,'importance'] = factor_importance_new
    st.session_state.df_factors.at[index_to_edit,'controllability'] = factor_controllability_new
    st.session_state.df_factors.at[index_to_edit,'strategic_importance'] = factor_strategic_new
    st.session_state.df_factors.at[index_to_edit,'uncertainty'] = factor_uncertainty_new
    st.session_state.df_factors.at[index_to_edit,'vulnerable_to_change'] = factor_vulnerable_new
    st.session_state.df_factors.at[index_to_edit,'resistant_to_change'] = factor_resistant_new
    st.session_state.df_factors.at[index_to_edit,'other_info'] = factor_info_new
    
# Delete existing factor
def delete_factor(index_to_delete, factor_name_to_delete):
    
    st.session_state.df_factors = st.session_state.df_factors.drop(index_to_delete)
    st.session_state.df_relationships.drop(index=st.session_state.df_relationships[st.session_state.df_relationships['from'] == factor_name_to_delete].index, inplace = True)
    st.session_state.df_relationships.drop(index=st.session_state.df_relationships[st.session_state.df_relationships['to'] == factor_name_to_delete].index, inplace = True)

# PLACEHOLDER — TBC
def update_domains_after_factor_delete():
    # check unique entries in df_factors.domain --> if domain ceases to exist delete it from df_domains
    pass

def plot_factors():
    
    G=nx.empty_graph()

    for index, row in st.session_state.df_factors.iterrows():
        if row['domain_id']==0: 
            size=40
            color='yellow'
        if row['domain_id']==1: 
            size=15
            color='cornflowerblue'

        G.add_node(row['factor_id'], label=row['long_name'], group=row['domain_id'], size=size, color=color)
        
    nt = net.Network(width='2000px', height='1800px', directed=True)
    nt.from_nx(G)
    nt.force_atlas_2based(gravity=-50)
    # nt.show_buttons(filter_=['physics'])
    nt.show("G_factors.html")
    HtmlFile = open('G_factors.html','r',encoding='utf-8')
    components.html(HtmlFile.read(),height=1800)


###############################
####### RELATIONSHIPS #########
###############################

# Load empty domains template
def load_relationships_template_table():
    
    if 'df_relationships' not in st.session_state:
        sheet_id = '1YDsfTegWgBnH4KB4FJ5g7fegaaKFFItTkuOF3Fbc3Vc'
        sheet_name = 'RELATIONSHIPS'
        url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
        df_relationships=pd.read_csv(url)
        df_relationships = df_relationships.loc[:, ~df_relationships.columns.str.contains('^Unnamed')]
        st.session_state.df_relationships = df_relationships
        
    
# Load factors from pre-filled Google Sheets template (input is sheet id)
def load_relationships(sheet_id):
    
    if st.session_state.language == 'English':
        sheet_name = 'RELATIONSHIPS'
    if st.session_state.language == 'Finnish':
        sheet_name = 'RELATIONSHIPS_fi_FINAL'
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    df_relationships=pd.read_csv(url)
    df_relationships = df_relationships.loc[:, ~df_relationships.columns.str.contains('^Unnamed')]
    st.session_state.df_relationships = df_relationships
    st.session_state.df_relationships['relationship_id'] = st.session_state.df_relationships.index
    st.session_state.df_relationships_S = st.session_state.df_relationships[st.session_state.df_relationships['strength'].isin(['strong'])]

def add_relationship(relationship_from,
                     relationship_to,
                     relationship_polarity,
                     relationship_strength,
                     relationship_temporality,
                     relationship_importance):
    
    from_id_lookup = st.session_state.df_factors[st.session_state.df_factors['long_name'] == relationship_from].factor_id.values[0]
    to_id_lookup = st.session_state.df_factors[st.session_state.df_factors['long_name'] == relationship_to].factor_id.values[0]
    
    st.session_state.relationships_counter = st.session_state.relationships_counter + 1
    
    new_relationship = {
        'relationship_id': st.session_state.relationships_counter, 
        'from': relationship_from,
        'to': relationship_to,
        'from_factor_id': from_id_lookup,
        'to_factor_id': to_id_lookup,
        'polarity': relationship_polarity,
        'strength': relationship_strength,
        'importance': relationship_temporality,
        'temporality': relationship_importance}
    
    st.session_state.df_relationships = st.session_state.df_relationships.append(new_relationship, ignore_index=True)    
    
def delete_relationship(index_to_delete):
    
    st.session_state.df_relationships = st.session_state.df_relationships.drop(index_to_delete)

def plot_relationships(CLD_rel_choice,CLD_isolates_choice):
    
    G=nx.empty_graph(create_using=nx.DiGraph())

    for index, row in st.session_state.df_factors.iterrows():
        if row['domain_id']==1:
            size=15
            color='cornflowerblue'
        if row['domain_id']==0: 
            size=40
            color='yellow'
        G.add_node(row['factor_id'], label=row['long_name'], group=row['domain_id'], size=size, color=color)

    if CLD_rel_choice == 'All':

        for index, row in st.session_state.df_relationships.iterrows():

            edge_from = row['from_factor_id']
            edge_to = row['to_factor_id']
            polarity = row['polarity']
            
            if polarity == 'positive': 
                title = 'positive'
                edge_color = 'lightgreen'  # choose your color for positive polarity
            if polarity == 'negative': 
                title = 'negative'
                edge_color = 'lightcoral'  # choose your color for negative polarity

            if row['strength']=='weak' or row['strength']=='nonlinear/unknown': 
                weight=0.1
                distance = 1/weight
            if row['strength']=='medium': 
                weight=4
                distance = 1/weight
            if row['strength']=='strong': 
                weight=10
                distance = 1/weight

            G.add_edge(edge_from, edge_to, weight=weight, hidden=False, arrowStrikethrough=False, color=edge_color)

    if CLD_rel_choice == 'Strong only':

        for index, row in st.session_state.df_relationships_S.iterrows():

            edge_from = row['from_factor_id']
            edge_to = row['to_factor_id']
            polarity = row['polarity']
            
            if polarity == 'positive': 
                title = 'positive'
                edge_color = 'lightgreen'  # choose your color for positive polarity
            if polarity == 'negative': 
                title = 'negative'
                edge_color = 'lightcoral'  # choose your color for negative polarity

            if row['strength']=='weak' or row['strength']=='nonlinear/unknown': 
                weight=0.1
                distance = 1/weight
            if row['strength']=='medium': 
                weight=4
                distance = 1/weight
            if row['strength']=='strong': 
                weight=10
                distance = 1/weight

            G.add_edge(edge_from, edge_to, weight=weight, hidden=False, arrowStrikethrough=False, color=edge_color)

    if CLD_isolates_choice == True:
        G.remove_nodes_from(list(nx.isolates(G)))

    # physics = st.checkbox('Show physics?')
    # if physics:
    #     nt = net.Network(width='1500px', height='1800px', directed=True)
    # else:
    #     nt = net.Network(width='2000px', height='1800px', directed=True)
    
    nt = net.Network(width='2500px', height='1800px', directed=True, select_menu=True, filter_menu=True, cdn_resources='in_line')
    nt.from_nx(G)
    nt.force_atlas_2based(gravity=-50)
    
    # if physics:
    #     nt.show_buttons(filter_=['physics'])
    nt.inherit_edge_colors(False)
#     nt.set_options("""
#     var options = {
#   "physics": {
#     "forceAtlas2Based": {
#       "springLength": 100
#     },
#     "minVelocity": 0.75,
#     "solver": "forceAtlas2Based"
#   }
# }
    
    
#     """)
    nt.show('G_factors_and_relationships.html')
    HtmlFile = open('G_factors_and_relationships.html','r',encoding='utf-8')
    components.html(HtmlFile.read(),height=1800)
    save_graph(G)
    return G   
    
###############################
########## ANALYSIS ###########
###############################

def load_centrality(G):
    df_nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    node_colors = df_nodes.color.to_list()
    df_nodes.drop(['size', 'color'], axis=1, inplace=True)
    
    in_degree_centrality_df = pd.DataFrame(nx.in_degree_centrality(G).items(), columns=["node", "in_degree_centrality"])
    
    out_degree_centrality_df = pd.DataFrame(nx.out_degree_centrality(G).items(), columns=["node", "out_degree_centrality"])
    
    closeness_centrality_df = pd.DataFrame(nx.closeness_centrality(G).items(), columns=["node", "closeness_centrality"])
    
    betweenness_centrality_df = pd.DataFrame(nx.betweenness_centrality(G).items(), columns=["node", "betweenness_centrality"])
    
    pagerank_centrality_df = pd.DataFrame(nx.pagerank(G).items(), columns=["node", "pagerank_centrality"])
    
    hub_scores, auth_scores = nx.hits(G)
    hub_centrality_df = pd.DataFrame(hub_scores.items(), columns=["node", "hub_centrality"])
    auth_centrality_df = pd.DataFrame(auth_scores.items(), columns=["node", "auth_centrality"])
    
    centrality_summary_df = in_degree_centrality_df\
    .merge(out_degree_centrality_df, on="node")\
    .merge(pagerank_centrality_df, on="node")\
    .merge(closeness_centrality_df, on="node")\
    .merge(betweenness_centrality_df, on="node")\
    .merge(hub_centrality_df, on="node")\
    .merge(auth_centrality_df, on="node")

    centrality_summary_df = df_nodes.join(centrality_summary_df).drop(['node'], axis=1)

    centrality_ranks = centrality_summary_df.rank(ascending=False, numeric_only=True, method="dense").astype(int, errors='ignore')
    
    # compute the average ranking using the above ranks
    average_ranks = pd.DataFrame(round(centrality_ranks.mean(axis=1)).astype(int), columns=["average_rank"])
    average_ranks.insert(loc=0, column='node', value=centrality_summary_df["label"])
    
    centrality_summary_df_styled = centrality_summary_df.style.background_gradient(subset=list(centrality_ranks.columns[1:]), cmap='PuBu_r').format(precision=2)
    centrality_summary_df_styled.to_html('centrality_summary_df_styled.html')
    
    centrality_ranks_df_styled = df_nodes.join(centrality_ranks.drop(['group'], axis=1), how='left').style.background_gradient(subset=list(centrality_ranks.columns[1:]),cmap='PuBu')
    centrality_ranks_df_styled.to_html('centrality_ranks_df_styled.html')

    average_ranks_df_styled = average_ranks.sort_values("average_rank").style.background_gradient(subset=["average_rank"], cmap='PuBu')
    average_ranks_df_styled.to_html('average_ranks_df_styled.html')
    
    return node_colors, centrality_summary_df
        
def draw(G, pos, measures, measure_name, ax):
    
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=250, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    # labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=ax)

    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()
    
    
def draw_centralities(G,node_colors):
    
    pos = nx.kamada_kawai_layout(G)
    
    f = plt.figure(constrained_layout=True, figsize=(25,20))
    gs = f.add_gridspec(4, 4)

    f_ax2 = f.add_subplot(gs[0:2,1:3])
    #f_ax2.set_title('table of factors')
    f_ax2.set_aspect('equal')

    f_ax3 = f.add_subplot(gs[2, 0])
    f_ax3.set_title('in-degree centrality')
    f_ax3.set_aspect('equal')

    f_ax4 = f.add_subplot(gs[2, 1])
    f_ax4.set_title('out-degree centrality')
    f_ax4.set_aspect('equal')

    f_ax5 = f.add_subplot(gs[2, 2])
    f_ax5.set_title('degree centrality')
    f_ax5.set_aspect('equal')

    f_ax6 = f.add_subplot(gs[2, 3])
    f_ax6.set_title('betweenness centrality')
    f_ax6.set_aspect('equal')

    f_ax7 = f.add_subplot(gs[3, 0])
    f_ax7.set_title('in-degree centrality')
    f_ax7.set_aspect('equal')

    f_ax8 = f.add_subplot(gs[3, 1])
    f_ax8.set_title('out-degree centrality')
    f_ax8.set_aspect('equal')

    f_ax9 = f.add_subplot(gs[3, 2])
    f_ax9.set_title('degree centrality')
    f_ax9.set_aspect('equal')

    f_ax10 = f.add_subplot(gs[3, 3])
    f_ax10.set_title('betweenness centrality')
    f_ax10.set_aspect('equal')
    
    nx.draw(G, pos, ax=f_ax2, with_labels=True, font_size=20, width=1, node_size=1000, node_color=node_colors)

    # in-degree centrality
    measure_name = 'In-Degree Centrality'
    measures = nx.in_degree_centrality(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax3, node_size=250, cmap=plt.cm.plasma, node_color=list(measures.values()), nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax3)
    f_ax3.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax3,location='right')

    # out-degree centrality
    measure_name = 'Out-Degree Centrality'
    measures = nx.out_degree_centrality(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax4, node_size=250, cmap=plt.cm.plasma, node_color=list(measures.values()), nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax4)
    f_ax4.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax4,location='right')

    # degree centrality
    measure_name = 'Degree Centrality'
    measures = nx.degree_centrality(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax5, node_size=250, cmap=plt.cm.plasma, node_color=list(measures.values()), nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax5)
    f_ax5.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax5,location='right')

    # betweenness centrality
    measure_name = 'Betweenness Centrality'
    measures = nx.betweenness_centrality(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax6, node_size=250, cmap=plt.cm.plasma, node_color=list(measures.values()), nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax6)
    f_ax6.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax6,location='right')

    # closeness centrality
    measure_name = 'Closeness Centrality'
    measures = nx.closeness_centrality(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax7, node_size=250, cmap=plt.cm.plasma, node_color=list(measures.values()), nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax7)
    f_ax7.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax7,location='right')

    # pagerank centrality
    measure_name = 'Pagerank Centrality'
    measures = nx.pagerank(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax8, node_size=250, cmap=plt.cm.plasma, node_color=list(measures.values()), nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax8)
    f_ax8.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax8,location='right')

    # hub centrality
    measure_name = 'Hub Centrality'
    h,a = nx.hits(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax9, node_size=250, cmap=plt.cm.plasma, node_color=list(h.values()), nodelist=h.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax9)
    f_ax9.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax9,location='right')

    # authority centrality
    measure_name = 'Authority Centrality'
    h,a = nx.hits(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax10, node_size=250, cmap=plt.cm.plasma, node_color=list(a.values()), nodelist=a.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax10)
    f_ax10.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax10,location='right')
    
    return f

def plot_centrality_archetypes(G):
        
        node_colors, centrality_summary_df = load_centrality(G)
        
        labels = centrality_summary_df['label'].values
        rows = centrality_summary_df.index.values
        centrality_names = ['IN','OUT','INFLUENCE','BROADCAST','BRIDGE','HUB','AUTHORITY']
        
        fig = make_subplots(rows=23, cols=4, specs=[[{'type': 'polar'}]*4]*23, subplot_titles=labels, vertical_spacing = 0.01)

        row_counter = 1
        column_counter = 1

        for count,value in enumerate(rows,start=1):

            modulus = count%4

            if modulus != 0:

                fig.add_trace(go.Scatterpolar(name = "", r=np.log(centrality_summary_df.iloc[value].values.tolist()[2:]), theta=centrality_names), row_counter, column_counter)

                column_counter = column_counter + 1

            if modulus == 0:

                fig.add_trace(go.Scatterpolar(name = "", r=np.log(centrality_summary_df.iloc[value].values.tolist()[2:]), theta=centrality_names), row_counter, column_counter)

                column_counter = 1
                row_counter = row_counter + 1

        fig.update_layout(
            polar_radialaxis_showticklabels=False,
            autosize=False,
            width=2000,
            height=13000,
            margin=dict(
                l=50,
                r=50,
                b=50,
                t=50,
                pad=0
            ))


        fig.update_polars(radialaxis=dict(range=[-9, 0]), radialaxis_showticklabels=False)
        fig.update_annotations(font_size=25)
        fig.update_traces(fill='toself')
        fig.layout.showlegend = False

        st.plotly_chart(fig, use_container_width=True)   

def control_centrality_single(G):
    
    A = nx.to_numpy_matrix(G).T
    N = G.number_of_nodes()
    factor_control_centralities = []
    
    for id in np.arange(0,N):
        
        B = np.zeros((N,1))
        B[id,0]=1
    
        C = control.ctrb(A,B)
        Cc = np.linalg.matrix_rank(C, tol=1.0e-10)
        cc = Cc/N
        
        factor_control_centralities.append(cc)
    
    cc_df = pd.DataFrame(factor_control_centralities, columns=['control_centrality'])
    
    df_nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    factor_control_centralities_df = df_nodes.join(cc_df).drop(['size','color'], axis=1)
    
    factor_control_centralities_df

    return factor_control_centralities_df.round(2)


def controllability_multiple(G,factors):
    
    A = nx.to_numpy_matrix(G).T          ##### <---- CHECK WHY TRANSPOSE IS NEEDED
    N = G.number_of_nodes()
    B = np.zeros((N,len(factors)))
    
    for i,factor_id in enumerate(factors):
        
        B[factor_id,i]=1
    
    C = control.ctrb(A,B)
    Cc = np.linalg.matrix_rank(C, tol=1.0e-10)
    cc = Cc/N

    return cc.round(2)


def plot_controllability_gauge(cc):
    
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = int(cc*100),
        number = { 'suffix': '%' },
        mode = "gauge+number",
        title = {'text': "What fraction of the system can these factors control?", 'font': {'size': 30}},
        delta = {'reference': 0},
        gauge = {'axis': {'range': [None, 100]}}))

    fig.update_layout(width=500, height=600)
    fig.update_layout(font=dict(size=30))
    st.plotly_chart(fig, use_container_width=True)        

def directed_to_bipartite(G_directed):

    G_bipartite=nx.empty_graph(create_using=nx.DiGraph())

    N = G_directed.number_of_nodes()

    bipartite_left_names = []
    bipartite_right_names = []

    for node in np.arange(0,N):

        bipartite_left = str(node) + '+'
        bipartite_right = str(node) + '-'

        bipartite_left_names.append(bipartite_left)
        bipartite_right_names.append(bipartite_right)

    G_bipartite.add_nodes_from(bipartite_left_names, bipartite=1)
    G_bipartite.add_nodes_from(bipartite_right_names, bipartite=0)

    edgelist_df = nx.to_pandas_edgelist(G_directed) 

    sources = edgelist_df.source.astype(str).tolist()
    targets = edgelist_df.target.astype(str).tolist()

    from_list = [source + "+" for source in sources]
    to_list = [target + "-" for target in targets]

    bipartite_edge_list = list(zip(from_list,to_list))

    G_bipartite.add_edges_from(bipartite_edge_list)
    
    left_nodes = {n for n, d in G_bipartite.nodes(data=True) if d["bipartite"] == 1}
    
    return G_bipartite, left_nodes

def N_D(G_directed, G_bipartite, left_nodes, algo):
    
    if algo == 1: matching_dict = nx.bipartite.hopcroft_karp_matching(G_bipartite, left_nodes)
    if algo == 2: matching_dict = nx.bipartite.eppstein_matching(G_bipartite, left_nodes)
    else: matching_dict = nx.bipartite.hopcroft_karp_matching(G_bipartite, left_nodes)

    a = list(matching_dict.keys())
    b = list(matching_dict.values())
    c = []
    c = a + b
    d = list(filter(lambda x: '-' in x, c))
    e = [s.strip('-') for s in d]
    f = list(set(e))
    g = list(map(int, f))
    
    matched_factors = sorted(g)
    all_factors = list(G_directed.nodes())
    unmatched_factors = sorted(list(set(all_factors) - set(matched_factors)))

    return len(unmatched_factors), unmatched_factors

def remove_bipartite_node(G_directed, G_bipartite, node):
    
    G_bipartite_trimmed = copy.deepcopy(G_bipartite)
    
    G_bipartite_trimmed.remove_node(str(node) + "+")
    G_bipartite_trimmed.remove_node(str(node) + "-")
    
    left_nodes_trimmed = {n for n, d in G_bipartite_trimmed.nodes(data=True) if d["bipartite"] == 1}
    
    return G_bipartite_trimmed, left_nodes_trimmed


def highlight_liu_classes(s):
    if s.Liu_class == 'neutral':
        return ['background-color: lightblue'] * len(s)
    if s.Liu_class == 'indispensable':
        return ['background-color: yellow'] * len(s)
    if s.Liu_class == 'dispensable':
        return ['background-color: lightgray'] * len(s)

def compute_liu_classes(G_directed, G_bipartite, ND):
    
    N = G_directed.number_of_nodes()

    liu_class_list = []

    for node in np.arange(0,N):

        G_directed_trimmed = copy.deepcopy(G_directed)
        G_directed_trimmed.remove_node(node)

        G_bipartite_trimmed, left_nodes_trimmed = remove_bipartite_node(G_directed, G_bipartite, node)

        ND_trimmed, unmatched_factors_trimmed = N_D(G_directed_trimmed, G_bipartite_trimmed, left_nodes_trimmed, node)

        if ND_trimmed > ND: liu_class = 'indispensable'
        if ND_trimmed == ND: liu_class = 'neutral'
        if ND_trimmed < ND: liu_class = 'dispensable'

        liu_class_list.append(liu_class)

    df_nodes = pd.DataFrame.from_dict(dict(G_directed.nodes(data=True)), orient='index')
    liu_class_summary_df = df_nodes.join(pd.DataFrame(liu_class_list, columns=['Liu_class'])).drop(['size', 'color', 'group'], axis=1)
    
    return liu_class_summary_df.style.apply(highlight_liu_classes, axis=1)


def highlight_jia_classes(s):
    if s.Jia_class == 'Intermittent':
        return ['background-color: lightblue'] * len(s)
    if s.Jia_class == 'Critical':
        return ['background-color: yellow'] * len(s)
    if s.Jia_class == 'Redundant':
        return ['background-color: lightgray'] * len(s)

def compute_jia_classes(G_directed,G_bipartite,ND):

    N = G_directed.number_of_nodes()
    nodes_in_all_MDS = list(node for node, in_degree in G_directed.in_degree() if in_degree == 0)
    never_MDS = 0
    some_MDS = 0
    all_MDS = 0
    jia_class_list = []

    for node in np.arange(0,N):

        G_directed_trimmed = copy.deepcopy(G_directed)
        G_bipartite_trimmed = copy.deepcopy(G_bipartite)

        G_bipartite_trimmed.remove_node(str(node) + "-")
        G_bipartite_trimmed.add_node(str(node) + "-", bipartite=0)

        left_nodes_trimmed = {n for n, d in G_bipartite_trimmed.nodes(data=True) if d["bipartite"] == 1}

        ND_trimmed, unmatched_factors_trimmed = N_D(G_directed_trimmed, G_bipartite_trimmed, left_nodes_trimmed, 1)

        if ND != ND_trimmed: 
            jia_class = 'Redundant'
            never_MDS+=1
        else: 
            jia_class = 'Intermittent'
            if (node in nodes_in_all_MDS): jia_class = 'Critical'
            some_MDS+=1

        jia_class_list.append(jia_class)

    df_nodes = pd.DataFrame.from_dict(dict(G_directed.nodes()), orient='index')
    jia_class_summary_df = df_nodes.join(pd.DataFrame(jia_class_list, columns=['Jia_class'])).drop(['size', 'color', 'group'], axis=1)
    
    return jia_class_summary_df

def compute_all_MIS(G_directed,jia_class_summary_df,ND):

    all_factors = list(G_directed.nodes())
    
    critical_list = jia_class_summary_df[jia_class_summary_df.Jia_class == 'Critical'].index.tolist()
    redundant_list = jia_class_summary_df[jia_class_summary_df.Jia_class == 'Redundant'].index.tolist()
    intermittent_list = jia_class_summary_df[jia_class_summary_df.Jia_class == 'Intermittent'].index.tolist()

    intermittent_candidate_combinations = list(itertools.combinations(intermittent_list, ND-len(critical_list)))

    MIS_list = []

    for i, combination in enumerate(intermittent_candidate_combinations):

        candidate_MIS = [] + critical_list + list(combination)

        A = nx.to_numpy_matrix(G_directed).T
        N = G_directed.number_of_nodes()
        B = np.zeros((N,len(candidate_MIS)))

        for i,factor_id in enumerate(candidate_MIS):

            B[factor_id,i]=1

        C = control.ctrb(A,B)
        Cc = np.linalg.matrix_rank(C,tol=1.0e-10)
        cc = Cc/N

        if cc == 1: MIS_list.append(candidate_MIS)


    df = pd.DataFrame(MIS_list)
    df
    
    MIS_column_headers = ['Factor']

    number_of_MIS = len(df.index)

    for MIS in np.arange(0,number_of_MIS):

        header = 'MIS_' + str(MIS + 1)
        MIS_column_headers.append(header)
        
    df_nodes = pd.DataFrame.from_dict(dict(G_directed.nodes(data=True)), orient='index')
    factor_names_list = df_nodes.label.to_list()
    MIS_df = pd.DataFrame(columns=MIS_column_headers)
    MIS_df.Factor = factor_names_list

    for index, row in df.iterrows():
    
        for factor_id in row.to_list():

            column = 'MIS_' + str(index + 1)
            MIS_df.loc[factor_id,column] = 'X'

        MIS_df = MIS_df.fillna('')

    return MIS_df



### Path Analysis — Intended and Unintended Consequences ###

def compute_path_polarity(G_directed, path):
    
    path_polarity = 1
    
    for source, target in zip(path, path[1:]):
        
        edge_polarity = G_directed.get_edge_data(source,target)['polarity']
        
        if edge_polarity == 1: path_polarity = path_polarity * 1
        if edge_polarity == -1: path_polarity = path_polarity * -1
                    
    return int(path_polarity)

def compute_icucpath_analysis(G_directed, source_index, target_index):

    intended_df = pd.DataFrame(columns=['Path','Polarity','Delay'])
    unintended_df = pd.DataFrame(columns=['Path','Polarity','Delay'])

    intended_df['Path']=intended_df['Path'].astype('object')
    unintended_df['Path']=unintended_df['Path'].astype('object')

    paths = list(nx.all_simple_paths(G_directed, source_index, target_index, cutoff=None))

    for path in paths:

        polarity = compute_path_polarity(G_directed, path)
        delay = ''

        if polarity == 1: 

            df_length = len(intended_df)  
            intended_df.loc[df_length] = [path, polarity, delay]  

        if polarity == -1: 

            df_length = len(unintended_df)  
            unintended_df.loc[df_length] = [path, polarity, delay]


    return intended_df, unintended_df


def path_to_sentence(G,path):
    
    i = 0
    
    st.write('**decoding path** ', str(path), ':')

    for source,target in zip(path, path[1:]):

        source_name = st.session_state.df_factors.loc[source].long_name
        target_name = st.session_state.df_factors.loc[target].long_name
        edge_polarity = compute_path_polarity(G, [source, target])

        if i == 0: cause = ' *increases* '
        else: cause = effect

        if edge_polarity == 1: 
            if cause == ' *increases* ': effect = ' *increases* '
            if cause == ' *decreases* ': effect = ' *decreases* '
        if edge_polarity == -1: 
            if cause == ' *increases* ': effect = ' *decreases* '
            if cause == ' *decreases* ': effect = ' *increases* '

        edge_statement = '- when ' + '**' + source_name + '**' + cause + '**' + target_name + '**' + effect

        i = i + 1    
        
        st.write(edge_statement)
    
def plot_icucpaths(G,path_intended,path_unintended):
    
        edges_ic = []
        edges_uc = []

        for source, target in zip(path_intended, path_intended[1:]):
            edges_ic.append((source,target))

        for source, target in zip(path_unintended, path_unintended[1:]):
            edges_uc.append((source,target)) 

        edges = edges_ic + edges_uc 

        G_sub = G.edge_subgraph(edges)

        nt = net.Network(width='1500px', height='1000px', directed=True)
        nt.from_nx(G_sub)
        nt.inherit_edge_colors(False)
        nt.show("icucpaths.html", notebook=False)
        HtmlFile = open('icucpaths.html','r',encoding='utf-8')
        components.html(HtmlFile.read(),height=700)


        
### Archetype Detection — Generic Structural Problems ###

def find_loops(G_directed, target_node):
    
    lst = sorted(nx.simple_cycles(G_directed))
    loops_that_contain_target_node = [loop for loop in lst if target_node in loop]
    
    return loops_that_contain_target_node

def close_loops(loops):
    
    closed_loops = []
    
    for i,loop in enumerate(loops):
        
        closed_loop = loop + [loop[0]]
        closed_loops.append(closed_loop)
    
    return closed_loops
        
def find_ADAS_archetypes(G_directed, target_node, bool_ic, bool_uc):
    
    archetypes_df = pd.DataFrame(columns=['ic_loop','uc_loop'])
    ic_list = []
    uc_list = []
    
    loops = find_loops(G_directed, target_node)
    closed_loops = close_loops(loops)
    
    for i,loop in enumerate(closed_loops):

        path_polarity = compute_path_polarity(G_directed, loop)
        
        if (path_polarity > 0) == bool_ic: ic_list.append(loop)
        if (path_polarity > 0) == bool_uc: uc_list.append(loop)
        
    for i,ic_loop in enumerate(ic_list):
        
        for j,uc_loop in enumerate(uc_list):
            
            intersection = False
            
            for k,n in enumerate(ic_loop):
                    
                exist_count = uc_loop.count(n)
                    
                if (n != target_node) and (exist_count > 0): 
                            
                    intersection = True
                    break
                            
            if intersection == False: 
                
                df_length = len(archetypes_df)  
                archetypes_df.loc[df_length] = [ic_loop, uc_loop]               

    archetypes_df = archetypes_df[['ic_loop','uc_loop']].astype('str')
    
    archetypes_df_unique = pd.DataFrame(np.sort(archetypes_df.values, axis=1), columns=archetypes_df.columns) \
    .value_counts().reset_index(name='counts').drop(['counts'], axis=1)                

    return archetypes_df        
        
def compute_archetype_dataframe(G,df):

    ic_df = pd.DataFrame(columns=['ic_Path','ic_Polarity','ic_Delay'])
    uc_df = pd.DataFrame(columns=['uc_Path','uc_Polarity','uc_Delay'])

    ic_df['ic_Path']=ic_df['ic_Path'].astype('object')
    uc_df['uc_Path']=uc_df['uc_Path'].astype('object')

    for index, row in df.iterrows():

        ic_path = ast.literal_eval(row['ic_loop'])
        uc_path = ast.literal_eval(row['uc_loop'])

        ic_path_polarity = compute_path_polarity(G, ic_path)
        uc_path_polarity = compute_path_polarity(G, uc_path)

        ic_path_delay = ''
        uc_path_delay = ''

        ic_df.loc[index] = [ic_path, ic_path_polarity, ic_path_delay]
        uc_df.loc[index] = [uc_path, uc_path_polarity, uc_path_delay]

    df_merged = pd.concat([ic_df, uc_df], axis=1)

    return df_merged

def save_graph(G):
    #nx.write_gpickle(G, "test.gpickle")
    return None