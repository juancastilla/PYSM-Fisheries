import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pyvis import network as net
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
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
from IPython.display import SVG
from sknetwork.hierarchy import LouvainHierarchy, Paris
from sknetwork.hierarchy import cut_straight, dasgupta_score, tree_sampling_divergence
from sknetwork.visualization import svg_graph, svg_bigraph, svg_dendrogram
import base64
import textwrap
import math
from sklearn.preprocessing import MinMaxScaler
import hiplot as hip
from deap import base, creator, tools, algorithms
import random
from plotapi import Chord
from PIL import Image
from scipy.stats import qmc, dirichlet
import SALib as sa
import seaborn as sns
from SALib.analyze import sobol
from SALib.sample import saltelli


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

### RENDER SVG ###

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

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

def plot_relationships(CLD_rel_choice,CLD_isolates_choice,mode):
    
    G=nx.empty_graph(create_using=nx.DiGraph())

    # st.dataframe(st.session_state.df_factors)

    for index, row in st.session_state.df_factors.iterrows():

        G.add_node(row['factor_id'], label=row['long_name'], domain=row['domain_id'])
    
    # for node in G.nodes(data=True):
    #     st.write(node)

    if CLD_rel_choice == 'All relationships':

        for index, row in st.session_state.df_relationships.iterrows():

            if row['strength']!='uncertain':

                edge_from = row['from_factor_id']
                edge_to = row['to_factor_id']
                polarity = row['polarity']
                
                if polarity == 'positive': 
                    title = 'positive'
                    edge_color = 'lightgreen'  # choose your color for positive polarity
                if polarity == 'negative': 
                    title = 'negative'
                    edge_color = 'lightcoral'  # choose your color for negative polarity

                if row['strength']=='weak': 
                    weight=0.1
                    distance = 1/weight
                    strength='weak'
                if row['strength']=='medium': 
                    weight=4
                    distance = 1/weight
                    strength='medium'
                if row['strength']=='strong': 
                    weight=10
                    distance = 1/weight
                    strength='strong'

                G.add_edge(edge_from, edge_to, weight=weight, hidden=False, arrowStrikethrough=False, color=edge_color, polarity=polarity, strength=strength, edge_value=weight)

    if CLD_rel_choice == 'Strong only':

        for index, row in st.session_state.df_relationships_S.iterrows():

            if row['strength']!='uncertain':

                edge_from = row['from_factor_id']
                edge_to = row['to_factor_id']
                polarity = row['polarity']
                
                if polarity == 'positive': 
                    title = 'positive'
                    edge_color = 'lightgreen'  # choose your color for positive polarity
                if polarity == 'negative': 
                    title = 'negative'
                    edge_color = 'lightcoral'  # choose your color for negative polarity

                if row['strength']=='weak': 
                    weight=0.1
                    distance = 1/weight
                    strength='weak'
                if row['strength']=='medium': 
                    weight=4
                    distance = 1/weight
                    strength='medium'
                if row['strength']=='strong': 
                    weight=10
                    distance = 1/weight
                    strength='strong'

                G.add_edge(edge_from, edge_to, weight=weight, hidden=False, arrowStrikethrough=False, color=edge_color, polarity=polarity, strength=strength)

    if CLD_isolates_choice == True:
        G.remove_nodes_from(list(nx.isolates(G)))
        # largest = max(nx.weakly_connected_components(G), key=len)
        # G = G.subgraph(largest) # largest connected component subgraph

    # physics = st.checkbox('Show physics?')
    # if physics:
    #     nt = net.Network(width='1500px', height='1800px', directed=True)
    # else:
    #     nt = net.Network(width='2000px', height='1800px', directed=True)
    
    if mode == 'display':

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




        # # Get the nodes in nt and G
        # nt_nodes = set(node['id'] for node in nt.nodes)
        # G_nodes = set(G.nodes)

        # # Find the nodes that are in nt but not in G
        # extra_nodes = nt_nodes - G_nodes

        # st.text(G_nodes)

        # # Remove the extra nodes from nt
        # for node in extra_nodes:
        #     nt.remove_node(node)

        # for node in nt.nodes:
        #     st.text(node)

        for node in nt.nodes:
            domain = node['domain']
            if domain == 0:
                node['color'] = 'yellow'
                node['size'] = 40
            elif domain == 1:
                node['color'] = 'cornflowerblue'
                node['size'] = 15
            elif domain == 2:
                node['color'] = 'magenta'
                node['size'] = 15
            elif domain == 3:
                node['color'] = 'lightcoral'
                node['size'] = 15
            elif domain == 4:
                node['color'] = 'orange'
                node['size'] = 15
            elif domain == 5:
                node['color'] = 'purple'
                node['size'] = 15

        # Save and read graph as HTML file (on Streamlit Sharing)
        try:
            path = './streamlit/html_files'
            nt.save_graph(f'{path}/pyvis_graph.html')
            HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')
            
            # Save and read graph as HTML file (locally)
        except:
            path = 'html_files'
            nt.save_graph(f'{path}/pyvis_graph.html')
            HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')

                # nt.show('G_factors_and_relationships.html')
                # HtmlFile = open('G_factors_and_relationships.html','r',encoding='utf-8')
        components.html(HtmlFile.read(),height=1800)
                # save_graph(G)

    return G   

def plot_relationships_submaps(SUBMAP_rel_choice, SUBMAP_steps_choice, selected_factor_id):
    
    G=nx.empty_graph(create_using=nx.DiGraph())

    for index, row in st.session_state.df_factors.iterrows():

        G.add_node(row['factor_id'], label=row['long_name'], domain=row['domain_id'])

    if SUBMAP_rel_choice == 'All relationships':

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

    if SUBMAP_rel_choice == 'Strong only':

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

    # Ensure selected_factor_id is an integer
    selected_factor_id = int(selected_factor_id)

    # Check if selected_factor_id exists in the graph
    if selected_factor_id in G.nodes:
        # Create a subgraph two steps upstream and downstream from the selected node
        G_sub = nx.ego_graph(G, selected_factor_id, radius=SUBMAP_steps_choice, undirected=True)
    else:
        st.error(f"Factor ID {selected_factor_id} does not exist in the graph.")
        return

    # if CLD_isolates_choice == True:
    #     G.remove_nodes_from(list(nx.isolates(G)))

    # physics = st.checkbox('Show physics?')
    # if physics:
    #     nt = net.Network(width='1500px', height='1800px', directed=True)
    # else:
    #     nt = net.Network(width='2000px', height='1800px', directed=True)
    
    nt = net.Network(width='2500px', height='1800px', directed=True, select_menu=True, filter_menu=True, cdn_resources='in_line')
    nt.from_nx(G_sub)
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

    for node in nt.nodes:

        if node['domain'] == 0:
            node['color'] = 'yellow'
            node['size'] = 40
        if node['domain'] == 1:
            node['color'] = 'cornflowerblue'
            node['size'] = 15
        if node['domain'] == 2:
            node['color'] = 'magenta'
            node['size'] = 15
        if node['domain'] == 3:
            node['color'] = 'lightcoral'
            node['size'] = 15
        if node['domain'] == 4:
            node['color'] = 'orange'
            node['size'] = 15
        if node['domain'] == 5:
            node['color'] = 'purple'
            node['size'] = 15

    # Save and read graph as HTML file (on Streamlit Sharing)
    try:
        path = './streamlit/html_files'
        nt.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')
        
    # Save and read graph as HTML file (locally)
    except:
        path = 'html_files'
        nt.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')

    components.html(HtmlFile.read(),height=1800)

    # nt.show('G_submap.html')
    # HtmlFile = open('G_submap.html','r',encoding='utf-8')
    # components.html(HtmlFile.read(),height=1800)
    # save_graph(G_sub)
    return G_sub   
    
def plot_relationships_interventionlab(CLD_rel_choice,CLD_isolates_choice,mode, domains_to_remove, intervenable_filter):
    
    G=nx.empty_graph(create_using=nx.DiGraph())

    # st.dataframe(st.session_state.df_factors)

    for index, row in st.session_state.df_factors.iterrows():

        G.add_node(row['factor_id'], label=row['long_name'], domain=row['domain_id'], intervenable=row['intervenable'])
    
    # for node in G.nodes(data=True):
    #     st.write(node)

    if CLD_rel_choice == 'All relationships':

        for index, row in st.session_state.df_relationships.iterrows():

            if row['strength']!='uncertain':

                edge_from = row['from_factor_id']
                edge_to = row['to_factor_id']
                polarity = row['polarity']
                
                if polarity == 'positive': 
                    title = 'positive'
                    edge_color = 'lightgreen'  # choose your color for positive polarity
                if polarity == 'negative': 
                    title = 'negative'
                    edge_color = 'lightcoral'  # choose your color for negative polarity

                if row['strength']=='weak': 
                    weight=0.1
                    distance = 1/weight
                    strength='weak'
                if row['strength']=='medium': 
                    weight=4
                    distance = 1/weight
                    strength='medium'
                if row['strength']=='strong': 
                    weight=10
                    distance = 1/weight
                    strength='strong'

                G.add_edge(edge_from, edge_to, weight=weight, hidden=False, arrowStrikethrough=False, color=edge_color, polarity=polarity, strength=strength, edge_value=weight)

    if CLD_rel_choice == 'Strong only':

        for index, row in st.session_state.df_relationships_S.iterrows():

            if row['strength']!='uncertain':

                edge_from = row['from_factor_id']
                edge_to = row['to_factor_id']
                polarity = row['polarity']
                
                if polarity == 'positive': 
                    title = 'positive'
                    edge_color = 'lightgreen'  # choose your color for positive polarity
                if polarity == 'negative': 
                    title = 'negative'
                    edge_color = 'lightcoral'  # choose your color for negative polarity

                if row['strength']=='weak': 
                    weight=0.1
                    distance = 1/weight
                    strength='weak'
                if row['strength']=='medium': 
                    weight=4
                    distance = 1/weight
                    strength='medium'
                if row['strength']=='strong': 
                    weight=10
                    distance = 1/weight
                    strength='strong'

                G.add_edge(edge_from, edge_to, weight=weight, hidden=False, arrowStrikethrough=False, color=edge_color, polarity=polarity, strength=strength)

    if CLD_isolates_choice == True:
        G.remove_nodes_from(list(nx.isolates(G)))
        # largest = max(nx.weakly_connected_components(G), key=len)
        # G = G.subgraph(largest) # largest connected component subgraph

    # physics = st.checkbox('Show physics?')
    # if physics:
    #     nt = net.Network(width='1500px', height='1800px', directed=True)
    # else:
    #     nt = net.Network(width='2000px', height='1800px', directed=True)

    nodes_to_remove = [node for node, attr in G.nodes(data=True) if attr['domain'] in domains_to_remove]
    G.remove_nodes_from(nodes_to_remove)

    if intervenable_filter == 'Intervenable Only':
        G.remove_nodes_from([node for node, attr in G.nodes(data=True) if attr['intervenable'] == 'no'])

    if mode == 'display':

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




        # # Get the nodes in nt and G
        # nt_nodes = set(node['id'] for node in nt.nodes)
        # G_nodes = set(G.nodes)

        # # Find the nodes that are in nt but not in G
        # extra_nodes = nt_nodes - G_nodes

        # st.text(G_nodes)

        # # Remove the extra nodes from nt
        # for node in extra_nodes:
        #     nt.remove_node(node)

        # for node in nt.nodes:
        #     st.text(node)

        for node in nt.nodes:
            domain = node['domain']
            if domain == 0:
                node['color'] = 'yellow'
                node['size'] = 40
            elif domain == 1:
                node['color'] = 'cornflowerblue'
                node['size'] = 15
            elif domain == 2:
                node['color'] = 'magenta'
                node['size'] = 15
            elif domain == 3:
                node['color'] = 'lightcoral'
                node['size'] = 15
            elif domain == 4:
                node['color'] = 'orange'
                node['size'] = 15
            elif domain == 5:
                node['color'] = 'purple'
                node['size'] = 15

        # Save and read graph as HTML file (on Streamlit Sharing)
        try:
            path = './streamlit/html_files'
            nt.save_graph(f'{path}/pyvis_graph.html')
            HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')
            
            # Save and read graph as HTML file (locally)
        except:
            path = 'html_files'
            nt.save_graph(f'{path}/pyvis_graph.html')
            HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')

                # nt.show('G_factors_and_relationships.html')
                # HtmlFile = open('G_factors_and_relationships.html','r',encoding='utf-8')
        components.html(HtmlFile.read(),height=1800)
                # save_graph(G)

    return G  

###############################
########## ANALYSIS ###########
###############################

def load_centrality(G):

    df_nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')

    in_degree_centrality_df = pd.DataFrame(nx.in_degree_centrality(G).items(), columns=["node", "in_degree"])
    out_degree_centrality_df = pd.DataFrame(nx.out_degree_centrality(G).items(), columns=["node", "out_degree"])

    closeness_centrality_df = pd.DataFrame(nx.closeness_centrality(G, distance='distance').items(), columns=["node", "closeness"])
    current_flow_closeness_centrality_df = pd.DataFrame(nx.current_flow_closeness_centrality(G.to_undirected(), weight='edge_value').items(), columns=["node", "flow_closeness"])

    betweenness_centrality_df = pd.DataFrame(nx.betweenness_centrality(G, weight='distance').items(), columns=["node", "betweenness"])
    current_flow_betweenness_centrality_df = pd.DataFrame(nx.current_flow_betweenness_centrality(G.to_undirected(), weight='edge_value').items(), columns=["node", "flow_betweenness"])

    pagerank_centrality_df = pd.DataFrame(nx.pagerank(G, weight='edge_value').items(), columns=["node", "pagerank"])

    centrality_summary_df =\
    in_degree_centrality_df\
    .merge(out_degree_centrality_df, on="node")\
    .merge(current_flow_closeness_centrality_df, on="node")\
    .merge(current_flow_betweenness_centrality_df, on="node")\
    .merge(pagerank_centrality_df, on="node")

    # Calculate Hub and Authority centralities
    hubs, authorities = nx.hits(G)

    # Convert them to DataFrames
    hub_centrality_df = pd.DataFrame(hubs.items(), columns=["node", "hub"])
    authority_centrality_df = pd.DataFrame(authorities.items(), columns=["node", "authority"])

    # Merge them into the centrality_summary_df
    centrality_summary_df = (
        centrality_summary_df
        .merge(hub_centrality_df, on="node")
        .merge(authority_centrality_df, on="node")
)

    centrality_summary_df.set_index('node', inplace=True)
    centrality_summary_df = df_nodes.join(centrality_summary_df)
    
    centrality_ranks = centrality_summary_df.rank(ascending=False, numeric_only=True, method="dense").astype(int, errors='ignore').drop('domain', axis=1)
    
    # compute the average ranking using the above ranks
    average_ranks = pd.DataFrame(round(centrality_ranks.mean(axis=1)).astype(int), columns=["average_rank"])
    average_ranks.insert(loc=0, column='node', value=centrality_summary_df["label"])
    
    centrality_summary_df = centrality_summary_df.round(4)
    centrality_summary_df_styled = centrality_summary_df.style.background_gradient(subset=list(centrality_ranks.columns[0:]), cmap='PuBu_r')
    # centrality_summary_df_styled.to_html('PULPO_Chile_centrality_summary_df_styled_ALL.html')
    
    centrality_ranks_df_styled = df_nodes.join(centrality_ranks, how='left').style.background_gradient(subset=list(centrality_ranks.columns[0:]),cmap='PuBu')
    # centrality_ranks_df_styled.to_html('PULPO_Chile_centrality_ranks_df_styled_ALL.html')

    average_ranks_df_styled = average_ranks.sort_values("average_rank").style.background_gradient(subset=["average_rank"], cmap='PuBu')
    # average_ranks_df_styled.to_html('PULPO_Chile_average_ranks_df_styled_ALL.html')
    
    centrality_ranks_df = df_nodes.join(centrality_ranks)
    
    return centrality_summary_df_styled, centrality_ranks_df_styled, average_ranks_df_styled, centrality_summary_df, centrality_ranks_df
      
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
    
def draw_centralities(G):
    
    pos = nx.kamada_kawai_layout(G)
    
    f = plt.figure(constrained_layout=False, figsize=(15,28))
    gs = f.add_gridspec(6, 2)

    f_ax2 = f.add_subplot(gs[0:2,0:2])
    #f_ax2.set_title('table of factors')
    f_ax2.set_aspect('equal')

    f_ax3 = f.add_subplot(gs[2, 0])
    f_ax3.set_title('in-degree centrality')
    f_ax3.set_aspect('equal')

    f_ax4 = f.add_subplot(gs[2, 1])
    f_ax4.set_title('out-degree centrality')
    f_ax4.set_aspect('equal')

    f_ax5 = f.add_subplot(gs[3, 0])
    f_ax5.set_title('degree centrality')
    f_ax5.set_aspect('equal')

    f_ax6 = f.add_subplot(gs[3, 1])
    f_ax6.set_title('betweenness centrality')
    f_ax6.set_aspect('equal')

    f_ax7 = f.add_subplot(gs[4, 0])
    f_ax7.set_title('in-degree centrality')
    f_ax7.set_aspect('equal')

    f_ax8 = f.add_subplot(gs[4, 1])
    f_ax8.set_title('out-degree centrality')
    f_ax8.set_aspect('equal')

    f_ax9 = f.add_subplot(gs[5, 0])
    f_ax9.set_title('degree centrality')
    f_ax9.set_aspect('equal')

    f_ax10 = f.add_subplot(gs[5, 1])
    f_ax10.set_title('betweenness centrality')
    f_ax10.set_aspect('equal')
    
    nx.draw(G, pos, ax=f_ax2, with_labels=True, font_size=10, width=1, node_size=500, font_color='w')

    # in-degree centrality
    measure_name = 'In-Degree Centrality'
    measures = nx.in_degree_centrality(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax3, node_size=250, cmap=plt.cm.plasma, node_color=list(measures.values()), nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax3)
    nx.draw_networkx_labels(G, pos, ax=f_ax3, font_color='w', font_size=8)
    f_ax3.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax3,location='right')

    # out-degree centrality
    measure_name = 'Out-Degree Centrality'
    measures = nx.out_degree_centrality(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax4, node_size=250, cmap=plt.cm.plasma, node_color=list(measures.values()), nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax4)
    nx.draw_networkx_labels(G, pos, ax=f_ax4, font_color='w', font_size=8)
    f_ax4.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax4,location='right')

    # degree centrality
    measure_name = 'Degree Centrality'
    measures = nx.degree_centrality(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax5, node_size=250, cmap=plt.cm.plasma, node_color=list(measures.values()), nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax5)
    nx.draw_networkx_labels(G, pos, ax=f_ax5, font_color='w', font_size=8)
    f_ax5.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax5,location='right')

    # betweenness centrality
    measure_name = 'Betweenness Centrality'
    measures = nx.betweenness_centrality(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax6, node_size=250, cmap=plt.cm.plasma, node_color=list(measures.values()), nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax6)
    nx.draw_networkx_labels(G, pos, ax=f_ax6, font_color='w', font_size=8)
    f_ax6.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax6,location='right')

    # closeness centrality
    measure_name = 'Closeness Centrality'
    measures = nx.closeness_centrality(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax7, node_size=250, cmap=plt.cm.plasma, node_color=list(measures.values()), nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax7)
    nx.draw_networkx_labels(G, pos, ax=f_ax7, font_color='w', font_size=8)
    f_ax7.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax7,location='right')

    # pagerank centrality
    measure_name = 'Pagerank Centrality'
    measures = nx.pagerank(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax8, node_size=250, cmap=plt.cm.plasma, node_color=list(measures.values()), nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax8)
    nx.draw_networkx_labels(G, pos, ax=f_ax8, font_color='w', font_size=8)
    f_ax8.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax8,location='right')

    # hub centrality
    measure_name = 'Hub Centrality'
    h,a = nx.hits(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax9, node_size=250, cmap=plt.cm.plasma, node_color=list(h.values()), nodelist=h.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax9)
    nx.draw_networkx_labels(G, pos, ax=f_ax9, font_color='w', font_size=8)
    f_ax9.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax9,location='right')

    # authority centrality
    measure_name = 'Authority Centrality'
    h,a = nx.hits(G)
    nodes = nx.draw_networkx_nodes(G, pos, ax=f_ax10, node_size=250, cmap=plt.cm.plasma, node_color=list(a.values()), nodelist=a.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, ax=f_ax10)
    nx.draw_networkx_labels(G, pos, ax=f_ax10, font_color='w', font_size=8)
    f_ax10.set_title(measure_name, size=16, weight='bold')
    plt.colorbar(nodes,ax=f_ax10,location='right')
    
    return f

def plot_centrality_archetypes(G):
        
        centrality_summary_df_styled, centrality_ranks_df_styled, average_ranks_df_styled, centrality_summary_df, centrality_ranks_df = load_centrality(G)

        # Get the column names excluding 'label'
        columns = [col for col in centrality_summary_df.columns if col not in ['label', 'domain']]

        # Create a multiselect widget for the columns
        selected_columns = st.multiselect('Select the centrality metrics to include', columns, default=columns)

        # Always include 'label' in the selected columns
        selected_columns = ['label'] + selected_columns

        # Filter the DataFrame based on the selected columns
        centrality_summary_df = centrality_summary_df[selected_columns]

        # Create a scaler object
        scaler = MinMaxScaler()

        # Fit and transform the data in the DataFrame (excluding the 'label' column)
        centrality_summary_df.iloc[:, 1:] = scaler.fit_transform(centrality_summary_df.iloc[:, 1:])
        # Number of variables we're plotting.
        num_vars = len(centrality_summary_df.columns) - 1  # Subtract 1 for the label column

        # Split the circle into even parts and save the angles
        # so we know where to put each axis.
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # The plot is a circle, so we need to "complete the loop"
        # and append the start to the end.
        angles += angles[:1]

        # Calculate number of rows needed for subplots
        num_rows = math.ceil(centrality_summary_df.shape[0] / 4)

        fig, axs = plt.subplots(nrows=num_rows, ncols=4, subplot_kw=dict(polar=True), figsize=(24, 3.5*num_rows))

        # Flatten the axis array and remove extra subplots
        axs = axs.flatten()[:centrality_summary_df.shape[0]]

        # Calculate the global y-limits
        ylim_global = (centrality_summary_df.iloc[:, 1:].min().min(), centrality_summary_df.iloc[:, 1:].max().max())

        for i, ax in enumerate(axs):
            row = centrality_summary_df.iloc[i, 1:].tolist()  # Exclude the label column
            row += row[:1]

            ax.fill(angles, row, color='blue', alpha=0.25)
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(centrality_summary_df.columns[1:])  # Exclude the label column
            ax.set_title(centrality_summary_df.iloc[i]['label'], fontsize=20, fontweight='bold')  # Set the title of the subplot to the label
            ax.set_ylim(ylim_global)  # Set the same y-limits for all subplots

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing

        # Save and read graph as png file (on cloud)
        try:
            path = './streamlit/static'
            fig.savefig(f'{path}/centrality_archetypes.png')   
            fig.savefig(f'{path}/centrality_archetypes.pdf')   
            image = Image.open(f'{path}/centrality_archetypes.png')
            st.image(image)    
            
        # Save and read graph as HTML file (locally)
        except:
            path = 'static'
            fig.savefig(f'{path}/centrality_archetypes.png')
            fig.savefig(f'{path}/centrality_archetypes.pdf')       
            image = Image.open(f'{path}/centrality_archetypes.png')
            st.image(image)


def control_centrality_single(G):
    
    A = nx.to_numpy_array(G).T
    N = G.number_of_nodes()
    factor_control_centralities = []
    
    for id in np.arange(0,N):
        
        B = np.zeros((N,1))
        B[id,0]=1
    
        C = control.ctrb(A,B)
        Cc = np.linalg.matrix_rank(C, tol=1.0e-2)
        cc = Cc/N
        
        factor_control_centralities.append(cc)
    
    cc_df = pd.DataFrame(factor_control_centralities, columns=['control_centrality'])
    
    df_nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index').reset_index()
    factor_control_centralities_df = df_nodes.join(cc_df)
    
    factor_control_centralities_df

    return factor_control_centralities_df.round(2)

def controllability_multiple(G,subfactors_df):

    factors = subfactors_df.index.tolist()
    st.text(factors)
    st.text(len(factors))
    
    A = nx.to_numpy_matrix(G).T         
    N = G.number_of_nodes()
    B = np.zeros((N,len(factors)))
    
    for i,factor in enumerate(factors):

        st.text(i)

        B[factor,i]=1
    
    C = control.ctrb(A,B)
    Cc = np.linalg.matrix_rank(C, tol=1.0e-2)
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

    node_df = pd.DataFrame.from_dict(dict(G_directed.nodes(data=True)), orient='index')
    node_list = node_df.index.tolist()

    for node in node_list:

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
    
    left_nodes = {n for n, d in G_bipartite.nodes(data=True) if d['bipartite'] == 1}
    
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

    node_df = pd.DataFrame.from_dict(dict(G_directed.nodes(data=True)), orient='index')
    node_list = node_df.index.tolist()

    liu_class_list = []

    for node in node_list:

        G_directed_trimmed = nx.DiGraph(G_directed)
        G_directed_trimmed.remove_node(node)

        G_bipartite_trimmed, left_nodes_trimmed = remove_bipartite_node(G_directed, G_bipartite, node)

        ND_trimmed, unmatched_factors_trimmed = N_D(G_directed_trimmed, G_bipartite_trimmed, left_nodes_trimmed, node)

        if ND_trimmed > ND: liu_class = 'indispensable'
        if ND_trimmed == ND: liu_class = 'neutral'
        if ND_trimmed < ND: liu_class = 'dispensable'

        liu_class_list.append(liu_class)

    liu_class_summary_df = node_df.join(pd.DataFrame(list(zip(liu_class_list,node_list)), columns=['Liu_class','node']).set_index('node'))

    df_mapping = pd.DataFrame({'class': ['indispensable','neutral','dispensable']})
    sort_mapping = df_mapping.reset_index().set_index('class')

    liu_class_summary_df['class_num'] = liu_class_summary_df['Liu_class'].map(sort_mapping['index'])
    liu_class_summary_df.sort_values('class_num')

    return liu_class_summary_df.sort_values('class_num')

def highlight_jia_classes(s):
    if s.Jia_class == 'Intermittent':
        return ['background-color: lightblue'] * len(s)
    if s.Jia_class == 'Critical':
        return ['background-color: yellow'] * len(s)
    if s.Jia_class == 'Redundant':
        return ['background-color: lightgray'] * len(s)

def compute_jia_classes(G_directed,G_bipartite,ND):

    node_df = pd.DataFrame.from_dict(dict(G_directed.nodes(data=True)), orient='index')
    node_list = node_df.index.tolist()

    nodes_in_all_MDS = list(node for node, in_degree in G_directed.in_degree() if in_degree == 0)
    never_MDS = 0
    some_MDS = 0
    all_MDS = 0
    jia_class_list = []

    for node in node_list:

        G_directed_trimmed = nx.DiGraph(G_directed)
        G_bipartite_trimmed = nx.DiGraph(G_bipartite)

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

    # df_nodes = pd.DataFrame.from_dict(dict(G_directed.nodes()), orient='index')
    jia_class_summary_df = node_df.join(pd.DataFrame(list(zip(jia_class_list,node_list)), columns=['Jia_class','node']).set_index('node'))
    jia_class_summary_df = jia_class_summary_df.sort_values('Jia_class')

    return jia_class_summary_df

def compute_all_MIS(G_directed,jia_class_summary_df,ND):

    all_factors = list(G_directed.nodes())
    
    critical_list = jia_class_summary_df[jia_class_summary_df.Jia_class == 'Critical'].index.tolist()
    redundant_list = jia_class_summary_df[jia_class_summary_df.Jia_class == 'Redundant'].index.tolist()
    intermittent_list = jia_class_summary_df[jia_class_summary_df.Jia_class == 'Intermittent'].index.tolist()

    intermittent_candidate_combinations = list(itertools.combinations(intermittent_list, ND-len(critical_list)))

    MIS_list = []
    f_list = pd.DataFrame(all_factors,columns=['factor_id'])

    for i, combination in enumerate(intermittent_candidate_combinations):

        candidate_MIS = [] + critical_list + list(combination)

        A = nx.to_numpy_matrix(G_directed).T
        N = G_directed.number_of_nodes()
        B = np.zeros((N,len(candidate_MIS)))

        for i,factor_id in enumerate(candidate_MIS):

            pos = f_list.index[f_list['factor_id'] == factor_id].tolist()[0]

            B[pos,i]=1

        C = control.ctrb(A,B)
        Cc = np.linalg.matrix_rank(C,tol=1.0e-20)
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

            pos = f_list.index[f_list['factor_id'] == factor_id].tolist()[0]

            column = 'MIS_' + str(index + 1)
            MIS_df.loc[pos,column] = 'X'

        MIS_df = MIS_df.fillna('')

    return MIS_df

### Path Analysis — Intended and Unintended Consequences ###

def compute_path_polarity(G_directed, path):
    
    path_polarity = 1
    
    for source, target in zip(path, path[1:]):
        
        edge_polarity = G_directed.get_edge_data(source,target)['polarity']
        
        if edge_polarity == 'positive': path_polarity = path_polarity * 1
        if edge_polarity == 'negative': path_polarity = path_polarity * -1
                    
    return int(path_polarity)

def icuc_heatmap(G_directed):

    ic_uc_df = pd.DataFrame(columns=['control_node','target_node','ic_paths', 'uc_paths'])

    all_factors = list(G_directed.nodes())

    for control_node in all_factors:
        
        for target_node in all_factors:
            
            if control_node != target_node:
                
                intended_df = pd.DataFrame(columns=['Path','Polarity','Delay'])
                unintended_df = pd.DataFrame(columns=['Path','Polarity','Delay'])

                paths = list(nx.all_simple_paths(G_directed, control_node, target_node, cutoff=None))
                
                for path in paths:

                    polarity = compute_path_polarity(G_directed, path)
                    delay = ''

                    if polarity == 1: 

                        df_length = len(intended_df)  
                        intended_df.loc[df_length] = [path, polarity, delay]  

                    if polarity == -1: 

                        df_length = len(unintended_df)  
                        unintended_df.loc[df_length] = [path, polarity, delay]
                        
                ic_uc_df_length = len(ic_uc_df)
                ic_uc_df.loc[ic_uc_df_length] = [control_node, target_node, len(intended_df), len(unintended_df)]  

    result_ic = ic_uc_df.pivot(index='control_node', columns='target_node', values='ic_paths').fillna(0)
    result_uc = ic_uc_df.pivot(index='control_node', columns='target_node', values='uc_paths').fillna(0)

    # Create a new DataFrame that identifies where both result_ic and result_uc have non-zero entries
    result_both = (result_ic.ne(0) & result_uc.ne(0)).astype(int)

    # Get the node pairs that have a 1 in the result_both matrix
    node_pairs_indices = list(zip(*np.where(result_both == 1)))

    # Convert indices to factor_id
    node_pairs = [(result_both.columns[i], result_both.index[j]) for i, j in node_pairs_indices]

  # Convert factor_id to names
    node_pairs_names = [(st.session_state.df_factors.loc[i, 'long_name'], st.session_state.df_factors.loc[j, 'long_name']) for i, j in node_pairs]

    return result_ic, result_uc, result_both, node_pairs_names

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

        nt = net.Network(width='1200px', height='600px', directed=True)
        nt.from_nx(G_sub)
        nt.inherit_edge_colors(False)

        # Save and read graph as HTML file (on Streamlit Sharing)
        try:
            path = './streamlit/html_files'
            nt.save_graph(f'{path}/icucpaths.html')
            HtmlFile = open(f'{path}/icucpaths.html','r', encoding='utf-8')

        # Save and read graph as HTML file (locally)
        except:
            path = 'html_files'
            nt.save_graph(f'{path}/icucpaths.html')
            HtmlFile = open(f'{path}/icucpaths.html','r', encoding='utf-8')

        components.html(HtmlFile.read(),height=600)

### Tradeoff Analysis — Interactive Parallel Coordinate Plot ###

def pcp_preprocess(PCP_rel_choice):

    df = pd.DataFrame()

    # #### REPLACE THSI WITH GLOBAL SHEET ID ###########
    # sheet_id = "1KyvP07oU4zuGlLQ61W12bSDDKyEtyFRJIthEPk0Iito"
    # sheet_name_factors = "factors"
    # sheet_name_relationships = "relationships"
    # url_factors = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name_factors}"
    # url_relationships = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name_relationships}"
    # ################################################

    df_factors = st.session_state.df_factors

    G = plot_relationships(PCP_rel_choice,True,'no_display') 
    largest = max(nx.weakly_connected_components(G), key=len)
    G = G.subgraph(largest)

    # Run and load centrality analyses
    centrality_summary_df_styled, centrality_ranks_df_styled, average_ranks_df_styled, centrality_summary_df, centrality_ranks_df = load_centrality(G)
    
    # Run and load control centralities
    
    factor_control_centralities_df  = control_centrality_single(G)

    # Run and load Jia analysis

    G_bipartite, left_nodes = directed_to_bipartite(G)
    ND, unmatched_factors = N_D(G, G_bipartite, left_nodes, 1)
    jia_class_summary_df = compute_jia_classes(G,G_bipartite,ND)

    # Run and load Liu analysis

    G_bipartite, left_nodes = directed_to_bipartite(G)
    ND, unmatched_factors = N_D(G, G_bipartite, left_nodes, 1)
    liu_class_summary_df = compute_liu_classes(G, G_bipartite,ND)

    # Join the dataframes   

    df=df_factors[['controllability','level of knowledge','predictability', 'measurability cost', 'intervenable', 'domain_id']].join(centrality_summary_df).join(factor_control_centralities_df['control_centrality']).join(liu_class_summary_df['Liu_class']).join(jia_class_summary_df['Jia_class'])

    df = df.rename(columns={'in_degree': 'in_degree', 'out_degree': 'out_degree', 'flow_closeness': 'closeness', 'flow_betweenness': 'betweenness', 'pagerank': 'pagerank', 'Liu_class': 'robust_control', 'Jia_class': 'global_control'})

    cols = [
    'label',
    'domain_id',
    'intervenable',
    'controllability',
    'level of knowledge',
    'predictability',
    'measurability cost',
    'in_degree',
    'out_degree',
    'closeness',
    'pagerank',
    'betweenness',
    'control_centrality',
    'robust_control',
    'global_control']

    cols.reverse()

    df = df[cols]

    # df.loc[(df['intervenable'] == 'no'), 'intervenable'] = '1_not_intervenable'
    # df.loc[(df['intervenable'] == 'yes'), 'intervenable'] = '2_intervenable'

    df.loc[(df['controllability'] == 'uncontrollable'), 'controllability'] = '0_uncontrollable'
    df.loc[(df['controllability'] == 'low'), 'controllability'] = '1_low'
    df.loc[(df['controllability'] == 'medium'), 'controllability'] = '2_medium'
    df.loc[(df['controllability'] == 'high'), 'controllability'] = '3_high'

    df.loc[(df['level of knowledge'] == 'low'), 'level of knowledge'] = '1_low'
    df.loc[(df['level of knowledge'] == 'medium'), 'level of knowledge'] = '2_medium'
    df.loc[(df['level of knowledge'] == 'high'), 'level of knowledge'] = '3_high'

    df.loc[(df['predictability'] == 'low'), 'predictability'] = '1_low'
    df.loc[(df['predictability'] == 'medium'), 'predictability'] = '2_medium'
    df.loc[(df['predictability'] == 'high'), 'predictability'] = '3_high'

    df.loc[(df['measurability cost'] == 'low'), 'measurability cost'] = '1_low'
    df.loc[(df['measurability cost'] == 'medium'), 'measurability cost'] = '2_medium'
    df.loc[(df['measurability cost'] == 'high'), 'measurability cost'] = '3_high'

    df.loc[(df['robust_control'] == 'dispensable'), 'robust_control'] = '1_dispensable'
    df.loc[(df['robust_control'] == 'neutral'), 'robust_control'] = '2_neutral'
    df.loc[(df['robust_control'] == 'indispensable'), 'robust_control'] = '3_indispensable'

    df.loc[(df['global_control'] == 'Redundant'), 'global_control'] = '1_redundant'
    df.loc[(df['global_control'] == 'Intermittent'), 'global_control'] = '2_intermittent'
    df.loc[(df['global_control'] == 'Critical'), 'global_control'] = '3_critical'

    df = df.dropna()

    # List of columns to change
    cols_to_change = ['global_control','robust_control','measurability cost','predictability','level of knowledge','controllability']

    # # Iterate over each specified column
    # for col in cols_to_change:
    #     df[col] = df[col].str.split('_').str[0].astype(int)

    # # Select numeric columns only
    # df_numeric = df.select_dtypes(include=[np.number])

    # # Initialize a scaler
    # scaler = MinMaxScaler()

    # # Fit and transform the data
    # df_normalized = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    # # Replace original numeric columns with normalized ones
    # df[df_normalized.columns] = df_normalized

    pcp_hiplot = hip.Experiment.from_dataframe(df)
    pcp_hiplot.to_html('PCP_Hiplot.html')
    HtmlFile = open('PCP_Hiplot.html','r',encoding='utf-8')
    components.html(HtmlFile.read(),height=1500)

    st.dataframe(df)

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

########################
### INTERVENTION LAB ###
########################

### DIFFUSION MODELS ###

def pulse_diffusion_network_model(G, initial_tokens, num_steps, df, log_scale=False, rolling_window=1, vertical=False, case=''):
   
    # Define the strength to weight mapping
    strength_to_weight = {'strong': 3, 'medium': 2, 'weak': 1}

    # Initialize the tokens at each node to zero
    tokens = {node: 0 for node in G.nodes}

    # Update the tokens for the nodes specified in the initial_tokens dictionary
    for node, token in initial_tokens.items():
        tokens[node] = token

    # Initialize a dictionary to store the token counts at each time step
    # Use the label attribute as the key
    token_counts_over_time = {G.nodes[node]['label']: [tokens[node]] for node in G.nodes}  # Include the initial token counts

    # Run the diffusion process for the specified number of steps
    for step in range(num_steps):
        # Calculate the total tokens to be passed from each node
        total_outgoing_weights = {node: 0 for node in G.nodes}
        for node in G.nodes:
            for neighbor in G.neighbors(node):
                edge_strength = G.edges[node, neighbor]['strength']
                edge_weight = strength_to_weight[edge_strength]
                total_outgoing_weights[node] += edge_weight

        # Calculate the new tokens at each node
        new_tokens = tokens.copy()  # Start with the current tokens
        for node in G.nodes:
            for neighbor in G.neighbors(node):
                edge_strength = G.edges[node, neighbor]['strength']
                edge_weight = strength_to_weight[edge_strength]
                tokens_passed = tokens[node] * (edge_weight / total_outgoing_weights[node])
                new_tokens[neighbor] += tokens_passed
                new_tokens[node] -= tokens_passed  # Redistribute the passed tokens back to the node

        # Update the tokens
        tokens = new_tokens

        # Store the token counts at this time step
        for node in G.nodes:
            token_counts_over_time[G.nodes[node]['label']].append(tokens[node])

    # Convert the token counts to a DataFrame and transpose it
    df_token_counts = pd.DataFrame(token_counts_over_time).T

    # Calculate the percentage of nodes with non-zero token counts
    non_zero_tokens_percentage = (df_token_counts.astype(bool).sum(axis=1) > 0).sum() / len(df_token_counts) * 100

    # Identify outcome and intervention nodes
    outcome_nodes = df[df['OUTCOME NODE']]['long_name'].tolist()
    intervention_nodes = df[df['TOKENS'] != 0]['long_name'].tolist()

    col1,col2 = st.columns(2)

    with col1:

        st.markdown("#### % System Controlability")

        # Create a gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=non_zero_tokens_percentage.mean(),
            number={'suffix': "%"},  # Add a '%' suffix to the number displayed
            gauge={'axis': {'range': [None, 100]}}
        ))

        # Display the gauge chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    with col2:
    
        st.markdown("#### Leverage (strength of causal effects) on outcome nodes")

        # Create a line plot for the outcome nodes
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for outcome_node in outcome_nodes:
            # Compute a moving average with a window size of 5
            smoothed_token_counts = df_token_counts.loc[outcome_node].rolling(window=rolling_window).mean()
            ax.plot(smoothed_token_counts, label=outcome_node)
        
        plt.xlabel('Time step', fontsize=20)
        plt.ylabel('Token count (~strength of causal effect)', fontsize=20)
        plt.legend(prop={'size': 20})
        plt.grid(True)

        # Save and read graph as png file (on cloud)
        try:
            path = './streamlit/static'
            fig.savefig(f'{path}/pulse_causal_effects_line.png')      
            image = Image.open(f'{path}/pulse_causal_effects_line.png')
            st.image(image)    
            
        # Save and read graph as HTML file (locally)
        except:
            path = 'static'
            fig.savefig(f'{path}/pulse_causal_effects_line.png')      
            image = Image.open(f'{path}/pulse_causal_effects_line.png')
            st.image(image)


    # Create a heatmap of the token counts
    st.markdown("#### Leverage (strength of causal effects) on all nodes (red = outcome nodes; blue = intervention nodes)")

    # Apply log scale if log_scale is True
    if log_scale:
        df_token_counts = np.log1p(df_token_counts)

    if st.session_state.fishery == 'Jumbo Flying Squid':
        fig, ax = plt.subplots(figsize=(12, 18))
    else:
        fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(df_token_counts, annot=True, fmt=".1f", cmap='magma', annot_kws={"size": 7}, cbar=False, ax=ax)

    # Highlight outcome nodes in red
    for label in ax.get_yticklabels():
        if label.get_text() in outcome_nodes:
            label.set_color('red')

    # Highlight nodes that have non-zero token counts (TOKENS column in df dataframe) in blue
    for label in ax.get_yticklabels():
        if label.get_text() in intervention_nodes:
            label.set_color('blue')

    plt.xlabel('Time step')
    plt.tight_layout()

    # Save and read graph as png file (on cloud)
    try:
        path = './streamlit/static'
        fig.savefig(f'{path}/pulse_causal_effects_heat.png')      
        image = Image.open(f'{path}/pulse_causal_effects_heat.png')
        left_co, right_co = st.columns([0.05,0.95])
        with right_co:
            st.image(image, use_column_width='auto')    
        
    # Save and read graph as HTML file (locally)
    except:
        path = 'static'
        fig.savefig(f'{path}/pulse_causal_effects_heat.png')      
        image = Image.open(f'{path}/pulse_causal_effects_heat.png')
        left_co, right_co = st.columns([0.05,0.95])
        with right_co:
            st.image(image, use_column_width='auto')    

    st.markdown("#### Leverage (cumulative strength of causal effects = total area under token count curves) on all nodes")

    # Create a line plot for the outcome nodes
    if st.session_state.fishery == 'Jumbo Flying Squid':
        fig, ax = plt.subplots(figsize=(15, 15))
    else:
        fig, ax = plt.subplots(figsize=(15, 10))

    # Plot a bar chart showing the total area under the df_token_counts curve for all nodes
    for index, value in enumerate(df_token_counts.sum(axis=1)):
        sorted_sums = df_token_counts.sum(axis=1).sort_values()
        sorted_indices = sorted_sums.index
        colors = ['red' if idx in outcome_nodes else 'blue' if idx in intervention_nodes else 'gray' for idx in sorted_indices]
        ax.barh(range(len(sorted_indices)), sorted_sums, color=colors, alpha=0.5)
        ax.set_yticks(range(len(sorted_indices)))
        ax.set_yticklabels(sorted_indices)
        for label in ax.get_yticklabels():
            label.set_color('red' if label.get_text() in outcome_nodes else 'blue' if label.get_text() in intervention_nodes else 'gray')
        ax.set_xlabel('Total Area Under Token Count Curve (~cumulative strength of causal effect)')
    plt.tight_layout()
    
    # Save and read graph as png file (on cloud)
    try:
        path = './streamlit/static'
        fig.savefig(f'{path}/pulse_causal_effects_bar.png')
        image = Image.open(f'{path}/pulse_causal_effects_bar.png')
        st.image(image, use_column_width="auto")

    # Save and read graph as HTML file (locally)
    except:
        path = 'static'
        fig.savefig(f'{path}/pulse_causal_effects_bar.png')
        image = Image.open(f'{path}/pulse_causal_effects_bar.png')
        st.image(image, use_column_width="auto")

     ### Plot individual effects for each node

    # Determine the number of rows for the subplots
    num_rows = int(np.ceil(len(df_token_counts) / 3))

    # Find the maximum value in the dataset
    max_value = df_token_counts.max().max()

    # Create a new figure with subplots
    fig, axs = plt.subplots(num_rows, 3, figsize=(18, 4*num_rows))

    # Flatten the axes array
    axs = axs.flatten()

    # Iterate over each row in the DataFrame
    total_cumulative_strength_t10 = df_token_counts.iloc[:, :10].sum().sum()
    total_cumulative_strength_t20 = df_token_counts.iloc[:, :20].sum().sum()
    total_cumulative_strength_t30 = df_token_counts.iloc[:, :30].sum().sum()

    # Collect percentages and node names for boxplot
    percentages_t10 = []
    percentages_t20 = []
    percentages_t30 = []
    node_names = []

    for ax, (index, row) in zip(axs, df_token_counts.iterrows()):
        ax.plot(row.rolling(window=rolling_window).mean())
        # Calculate cumulative strengths at different timesteps
        cumulative_strength_t10 = row.iloc[:10].sum()
        cumulative_strength_t20 = row.iloc[:20].sum()
        cumulative_strength_t30 = row.iloc[:30].sum()
        # Calculate percentages of cumulative strengths
        percentage_t10 = (cumulative_strength_t10 / total_cumulative_strength_t10) * 100
        percentage_t20 = (cumulative_strength_t20 / total_cumulative_strength_t20) * 100
        percentage_t30 = (cumulative_strength_t30 / total_cumulative_strength_t30) * 100

        # Append percentages and node names for boxplot
        percentages_t10.append(percentage_t10)
        percentages_t20.append(percentage_t20)
        percentages_t30.append(percentage_t30)
        node_names.append(index)

        # Display the cumulative strengths and percentages on the plot
        ax.text(0.5, 0.98, f'Cumulative Strength t=10: {cumulative_strength_t10:.1f} ({percentage_t10:.1f}%)', transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.5, 0.88, f'Cumulative Strength t=20: {cumulative_strength_t20:.1f} ({percentage_t20:.1f}%)', transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.5, 0.78, f'Cumulative Strength t=30: {cumulative_strength_t30:.1f} ({percentage_t30:.1f}%)', transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        if index in outcome_nodes:
            ax.set_title(index, fontsize=20, color='red')  # Increase plot title font and set color to red
        if index in intervention_nodes:
            ax.set_title(index, fontsize=20, color='blue')
            # Get the number of tokens for the current node
            num_tokens = df[df['long_name'] == index]['TOKENS'].values[0]
            # Add the number of tokens as a text box in the subplot
            ax.text(0.35, 0.9, f'Tokens: {num_tokens}', transform=ax.transAxes, fontsize=24, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        else:
            ax.set_title(index, fontsize=20)  # Increase plot title font
        ax.set_xlabel('Time step')
        ax.set_ylabel('Token count (~strength of causal effect)')
        ax.set_ylim([0, max_value])  # Set the y-axis limits

    # Remove unused subplots
    for ax in axs[len(df_token_counts):]:
        fig.delaxes(ax)

    # Adjust the layout
    plt.tight_layout()

    # Plot boxplots for the percentages using Plotly

    # Create a DataFrame with the data
    data = {'Node': node_names, 't=10': percentages_t10, 't=20': percentages_t20, 't=30': percentages_t30}
    df = pd.DataFrame(data)

    # Display the DataFrame
    st.dataframe(df)

    # Add a download button
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="investment_efficiencies.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Plot the boxplots
    fig2 = go.Figure()
    fig2.add_trace(go.Box(y=percentages_t10, name='t=10', text=node_names, hoverinfo='text+y'))
    fig2.add_trace(go.Box(y=percentages_t20, name='t=20', text=node_names, hoverinfo='text+y'))
    fig2.add_trace(go.Box(y=percentages_t30, name='t=30', text=node_names, hoverinfo='text+y'))
    fig2.update_layout(title='Investment Efficiencies', yaxis_title='Investment Efficiency (%)')
    st.plotly_chart(fig2, use_container_width=True)

    # Save and read graph as png file (on cloud)
    try:
        path = './streamlit/static'
        fig.savefig(f'{path}/pulse_causal_effects_lines_all.png')
        image = Image.open(f'{path}/pulse_causal_effects_lines_all.png')
        st.image(image, use_column_width=True)
        
    # Save and read graph as HTML file (locally)
    except:
        path = 'static'
        fig.savefig(f'{path}/pulse_causal_effects_lines_all.png')
        image = Image.open(f'{path}/pulse_causal_effects_lines_all.png')
        st.image(image, use_column_width=True)

    return tokens

def flow_diffusion_network_model(G, token_injection_rate, num_steps, df, log_scale=False, rolling_window=1):

    # Define the strength to weight mapping
    strength_to_weight = {'strong': 3, 'medium': 2, 'weak': 1}

    intervention_nodes = df[df['TOKENS'] != 0]['long_name'].tolist()

    # Initialize the tokens at each node to the injection rate or zero if not specified
    tokens = {node: token_injection_rate.get(node, 0) for node in G.nodes}
    
    # Initialize a dictionary to store the token counts at each time step
    # Use the label attribute as the key
    token_counts_over_time = {G.nodes[node]['label']: [tokens[node]] for node in G.nodes}  # Include the initial token counts

    # Run the diffusion process for the specified number of steps
    for step in range(num_steps):
        # Calculate the total tokens to be passed from each node
        total_outgoing_tokens = {node: 0 for node in G.nodes}
        for node in G.nodes:
            for neighbor in G.neighbors(node):
                edge_strength = G.edges[node, neighbor]['strength']
                edge_weight = strength_to_weight[edge_strength]
                total_outgoing_tokens[node] += edge_weight

        # Calculate the new tokens at each node
        new_tokens = tokens.copy()  # Start with the current tokens
        for node in G.nodes:
            for neighbor in G.neighbors(node):
                edge_strength = G.edges[node, neighbor]['strength']
                edge_weight = strength_to_weight[edge_strength]
                tokens_passed = tokens[node] * (edge_weight / total_outgoing_tokens[node])
                new_tokens[neighbor] += tokens_passed
                if node not in intervention_nodes:  # Only subtract tokens if the node is not an intervention node
                    new_tokens[node] -= tokens_passed

        # Update the tokens
        tokens = new_tokens

        # Inject tokens at the specified nodes at a fixed rate
        for node, rate in token_injection_rate.items():
            tokens[node] += rate

        # Store the token counts at this time step
        for node in G.nodes:
            token_counts_over_time[G.nodes[node]['label']].append(tokens[node])

    # Convert the token counts to a DataFrame and transpose it
    df_token_counts = pd.DataFrame(token_counts_over_time).T

    # Calculate the percentage of nodes with non-zero token counts
    non_zero_tokens_percentage = (df_token_counts.astype(bool).sum(axis=1) > 0).sum() / len(df_token_counts) * 100

    # Identify outcome and intervention nodes
    outcome_nodes = df[df['OUTCOME NODE']]['long_name'].tolist()
    intervention_nodes = df[df['TOKENS'] != 0]['long_name'].tolist()

    col1,col2 = st.columns(2)

    with col1:

        st.markdown("#### % System Controlability")

        # Create a gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=non_zero_tokens_percentage.mean(),
            number={'suffix': "%"},  # Add a '%' suffix to the number displayed
            gauge={'axis': {'range': [None, 100]}}
        ))

        # Display the gauge chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    with col2:
    
        st.markdown("#### Leverage (strength of causal effects) on outcome nodes")

        # Create a line plot for the outcome nodes
        fig, ax = plt.subplots(figsize=(12, 8))
        for outcome_node in outcome_nodes:
            # Compute a moving average with a window size of 5
            smoothed_token_counts = df_token_counts.loc[outcome_node].rolling(rolling_window).mean()
            ax.plot(smoothed_token_counts, label=outcome_node)
        plt.xlabel('Time step', fontsize=20)
        plt.ylabel('Token count (~strength of causal effect)', fontsize=20)
        plt.legend(prop={'size': 20})
        plt.grid(True)

        # Save and read graph as png file (on cloud)
        try:
            path = './streamlit/static'
            fig.savefig(f'{path}/flow_causal_effects_line.png')      
            image = Image.open(f'{path}/flow_causal_effects_line.png')
            st.image(image)    
            
        # Save and read graph as HTML file (locally)
        except:
            path = 'static'
            fig.savefig(f'{path}/flow_causal_effects_line.png')      
            image = Image.open(f'{path}/flow_causal_effects_line.png')
            st.image(image)


    # Create a heatmap of the token counts

    st.markdown("#### Leverage (strength of causal effects) on all nodes (red = outcome nodes; blue = intervention nodes)")

    # Apply log scale if log_scale is True
    if log_scale:
        df_token_counts = np.log1p(df_token_counts)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(df_token_counts, annot=True, fmt=".1f", cmap='magma', annot_kws={"size": 5}, cbar=False)

    # Highlight outcome nodes in red
    for label in ax.get_yticklabels():
        if label.get_text() in outcome_nodes:
            label.set_color('red')

    # Highlight nodes that have non-zero token counts (TOKENS column in df dataframe) in blue
    for label in ax.get_yticklabels():
        if label.get_text() in intervention_nodes:
            label.set_color('blue')

    plt.xlabel('Time step')
    plt.tight_layout()

    # Save and read graph as png file (on cloud)
    try:
        path = './streamlit/static'
        fig.savefig(f'{path}/flow_causal_effects_heat.png')      
        image = Image.open(f'{path}/flow_causal_effects_heat.png')
        left_co, right_co = st.columns([0.05,0.95])
        with right_co:
            st.image(image, use_column_width='auto')    
        
    # Save and read graph as HTML file (locally)
    except:
        path = 'static'
        fig.savefig(f'{path}/flow_causal_effects_heat.png')      
        image = Image.open(f'{path}/flow_causal_effects_heat.png')
        left_co, right_co = st.columns([0.05,0.95])
        with right_co:
            st.image(image, use_column_width='auto')   

    
    st.markdown("#### Leverage (cumulative strength of causal effects = total area under token count curves) on all nodes")

    # Create a line plot for the outcome nodes
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot a bar chart showing the total area under the df_token_counts curve for all nodes
    for index, value in enumerate(df_token_counts.sum(axis=1)):
        sorted_sums = df_token_counts.sum(axis=1).sort_values()
        sorted_indices = sorted_sums.index
        colors = ['red' if idx in outcome_nodes else 'blue' if idx in intervention_nodes else 'gray' for idx in sorted_indices]
        ax.barh(range(len(sorted_indices)), sorted_sums, color=colors, alpha=0.5)
        ax.set_yticks(range(len(sorted_indices)))
        ax.set_yticklabels(sorted_indices)
        for label in ax.get_yticklabels():
            label.set_color('red' if label.get_text() in outcome_nodes else 'blue' if label.get_text() in intervention_nodes else 'gray')
        ax.set_xlabel('Total Area Under Token Count Curve (~cumulative strength of causal effect)')
    plt.tight_layout()
    
    # Save and read graph as png file (on cloud)
    try:
        path = './streamlit/static'
        fig.savefig(f'{path}/pulse_causal_effects_bar.png')
        image = Image.open(f'{path}/pulse_causal_effects_bar.png')
        st.image(image, use_column_width="auto")

    # Save and read graph as HTML file (locally)
    except:
        path = 'static'
        fig.savefig(f'{path}/pulse_causal_effects_bar.png')
        image = Image.open(f'{path}/pulse_causal_effects_bar.png')
        st.image(image, use_column_width="auto")
    
    ### Plot individual effects for each node

    # Determine the number of rows for the subplots
    num_rows = int(np.ceil(len(df_token_counts) / 3))

    # Find the maximum value in the dataset
    max_value = df_token_counts.max().max()

    # Create a new figure with subplots
    fig, axs = plt.subplots(num_rows, 3, figsize=(18, 4*num_rows))

    # Flatten the axes array
    axs = axs.flatten()

    # Iterate over each row in the DataFrame
    for ax, (index, row) in zip(axs, df_token_counts.iterrows()):
        ax.plot(row.rolling(window=rolling_window).mean())
        total_area = row.sum()
        ax.text(0.5, 0.98, f'Cumulative Strength of Effect: {total_area:.1f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        if index in outcome_nodes:
            ax.set_title(index, fontsize=20, color='red')  # Increase plot title font and set color to red
        if index in intervention_nodes:
            ax.set_title(index, fontsize=20, color='blue')
            # Get the number of tokens for the current node
            num_tokens = df[df['long_name'] == index]['TOKENS'].values[0]
            # Add the number of tokens as a text box in the subplot
            ax.text(0.35, 0.9, f'Tokens: {num_tokens}', transform=ax.transAxes, fontsize=24, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        else:
            ax.set_title(index, fontsize=20)  # Increase plot title font
        ax.set_xlabel('Time step')
        ax.set_ylabel('Token count (~strength of causal effect)')
        ax.set_ylim([0, max_value])  # Set the y-axis limits

    # Remove unused subplots
    for ax in axs[len(df_token_counts):]:
        fig.delaxes(ax)

    # Adjust the layout
    plt.tight_layout()

    # Save and read graph as png file (on cloud)
    try:
        path = './streamlit/static'
        fig.savefig(f'{path}/flow_causal_effects_lines_all.png')      
        image = Image.open(f'{path}/flow_causal_effects_lines_all.png')
        st.image(image, use_column_width=True)  
        
    # Save and read graph as HTML file (locally)
    except:
        path = 'static'
        fig.savefig(f'{path}/flow_causal_effects_lines_all.png')      
        image = Image.open(f'{path}/flow_causal_effects_lines_all.png')
        st.image(image, use_column_width=True)   

    return tokens


def diffusion_model_compare(G, token_dict_1, token_dict_2, diffusion_model, time_horizon, edited_df_1, edited_df_2, log_scale, rolling_window):

    # Run the diffusion model for both intervention packages

    # Define the strength to weight mapping
    strength_to_weight = {'strong': 3, 'medium': 2, 'weak': 1}

    if diffusion_model == 'one-time investment':

        ### CASE 1 ###

        # Initialize the tokens at each node to zero
        tokens_1 = {node: 0 for node in G.nodes}

        # Update the tokens for the nodes specified in the initial_tokens dictionary
        for node, token in token_dict_1.items():
            tokens_1[node] = token

        # Initialize a dictionary to store the token counts at each time step
        # Use the label attribute as the key
        token_counts_over_time_1 = {G.nodes[node]['label']: [tokens_1[node]] for node in G.nodes}  # Include the initial token counts

        # Run the diffusion process for the specified number of steps
        for step in range(time_horizon):
            # Calculate the total tokens to be passed from each node
            total_outgoing_weights_1 = {node: 0 for node in G.nodes}
            for node in G.nodes:
                for neighbor in G.neighbors(node):
                    edge_strength = G.edges[node, neighbor]['strength']
                    edge_weight = strength_to_weight[edge_strength]
                    total_outgoing_weights_1[node] += edge_weight

            # Calculate the new tokens at each node
            new_tokens_1 = tokens_1.copy()  # Start with the current tokens
            for node in G.nodes:
                for neighbor in G.neighbors(node):
                    edge_strength = G.edges[node, neighbor]['strength']
                    edge_weight = strength_to_weight[edge_strength]
                    tokens_passed = tokens_1[node] * (edge_weight / total_outgoing_weights_1[node])
                    new_tokens_1[neighbor] += tokens_passed
                    new_tokens_1[node] -= tokens_passed  # Redistribute the passed tokens back to the node

            # Update the tokens
            tokens_1 = new_tokens_1

            # Store the token counts at this time step
            for node in G.nodes:
                token_counts_over_time_1[G.nodes[node]['label']].append(tokens_1[node])

        # Convert the token counts to a DataFrame and transpose it
        df_token_counts_1 = pd.DataFrame(token_counts_over_time_1).T

        # Calculate the percentage of nodes with non-zero token counts
        non_zero_tokens_percentage_1 = (df_token_counts_1.astype(bool).sum(axis=1) > 0).sum() / len(df_token_counts_1) * 100

        # Identify outcome and intervention nodes
        outcome_nodes_1 = edited_df_1[edited_df_1['OUTCOME NODE']]['long_name'].tolist()
        intervention_nodes_1 = edited_df_1[edited_df_1['TOKENS'] != 0]['long_name'].tolist()


        ### CASE 2 ###

        # Initialize the tokens at each node to zero
        tokens_2 = {node: 0 for node in G.nodes}

        # Update the tokens for the nodes specified in the initial_tokens dictionary
        for node, token in token_dict_2.items():
            tokens_2[node] = token

        # Initialize a dictionary to store the token counts at each time step
        # Use the label attribute as the key
        token_counts_over_time_2 = {G.nodes[node]['label']: [tokens_2[node]] for node in G.nodes}  # Include the initial token counts

        # Run the diffusion process for the specified number of steps
        for step in range(time_horizon):
            # Calculate the total tokens to be passed from each node
            total_outgoing_weights_2 = {node: 0 for node in G.nodes}
            for node in G.nodes:
                for neighbor in G.neighbors(node):
                    edge_strength = G.edges[node, neighbor]['strength']
                    edge_weight = strength_to_weight[edge_strength]
                    total_outgoing_weights_2[node] += edge_weight

            # Calculate the new tokens at each node
            new_tokens_2 = tokens_2.copy()  # Start with the current tokens
            for node in G.nodes:
                for neighbor in G.neighbors(node):
                    edge_strength = G.edges[node, neighbor]['strength']
                    edge_weight = strength_to_weight[edge_strength]
                    tokens_passed = tokens_2[node] * (edge_weight / total_outgoing_weights_2[node])
                    new_tokens_2[neighbor] += tokens_passed
                    new_tokens_2[node] -= tokens_passed  # Redistribute the passed tokens back to the node

            # Update the tokens
            tokens_2 = new_tokens_2

            # Store the token counts at this time step
            for node in G.nodes:
                token_counts_over_time_2[G.nodes[node]['label']].append(tokens_2[node])

        # Convert the token counts to a DataFrame and transpose it
        df_token_counts_2 = pd.DataFrame(token_counts_over_time_2).T

        # Calculate the percentage of nodes with non-zero token counts
        non_zero_tokens_percentage_2 = (df_token_counts_2.astype(bool).sum(axis=1) > 0).sum() / len(df_token_counts_2) * 100

        # Identify outcome and intervention nodes
        outcome_nodes_2 = edited_df_2[edited_df_2['OUTCOME NODE']]['long_name'].tolist()
        intervention_nodes_2 = edited_df_2[edited_df_2['TOKENS'] != 0]['long_name'].tolist()


    if diffusion_model == 'continuous investment':

        ### CASE 1 ###

        # Identify outcome and intervention nodes
        outcome_nodes_1 = edited_df_1[edited_df_1['OUTCOME NODE']]['long_name'].tolist()
        intervention_nodes_1 = edited_df_1[edited_df_1['TOKENS'] != 0]['long_name'].tolist()  

        # Initialize the tokens at each node to the injection rate or zero if not specified
        tokens_1 = {node: token_dict_1.get(node, 0) for node in G.nodes}
        
        # Initialize a dictionary to store the token counts at each time step
        # Use the label attribute as the key
        token_counts_over_time_1 = {G.nodes[node]['label']: [tokens_1[node]] for node in G.nodes}  # Include the initial token counts

        # Run the diffusion process for the specified number of steps
        for step in range(time_horizon):
            # Calculate the total tokens to be passed from each node
            total_outgoing_tokens_1 = {node: 0 for node in G.nodes}
            for node in G.nodes:
                for neighbor in G.neighbors(node):
                    edge_strength = G.edges[node, neighbor]['strength']
                    edge_weight = strength_to_weight[edge_strength]
                    total_outgoing_tokens_1[node] += edge_weight

            # Calculate the new tokens at each node
            new_tokens_1 = tokens_1.copy()  # Start with the current tokens
            for node in G.nodes:
                for neighbor in G.neighbors(node):
                    edge_strength = G.edges[node, neighbor]['strength']
                    edge_weight = strength_to_weight[edge_strength]
                    tokens_passed = tokens_1[node] * (edge_weight / total_outgoing_tokens_1[node])
                    new_tokens_1[neighbor] += tokens_passed
                    if node not in intervention_nodes_1:  # Only subtract tokens if the node is not an intervention node
                        new_tokens_1[node] -= tokens_passed

            # Update the tokens
            tokens_1 = new_tokens_1

            # Inject tokens at the specified nodes at a fixed rate
            for node, rate in token_dict_1.items():
                tokens_1[node] += rate

            # Store the token counts at this time step
            for node in G.nodes:
                token_counts_over_time_1[G.nodes[node]['label']].append(tokens_1[node])

        # Convert the token counts to a DataFrame and transpose it
        df_token_counts_1 = pd.DataFrame(token_counts_over_time_1).T

        # Calculate the percentage of nodes with non-zero token counts
        non_zero_tokens_percentage_1 = (df_token_counts_1.astype(bool).sum(axis=1) > 0).sum() / len(df_token_counts_1) * 100

        
        ### CASE 2 ###

         # Identify outcome and intervention nodes
        outcome_nodes_2 = edited_df_2[edited_df_2['OUTCOME NODE']]['long_name'].tolist()
        intervention_nodes_2 = edited_df_2[edited_df_2['TOKENS'] != 0]['long_name'].tolist()  

        # Initialize the tokens at each node to the injection rate or zero if not specified
        tokens_2 = {node: token_dict_2.get(node, 0) for node in G.nodes}
        
        # Initialize a dictionary to store the token counts at each time step
        # Use the label attribute as the key
        token_counts_over_time_2 = {G.nodes[node]['label']: [tokens_2[node]] for node in G.nodes}  # Include the initial token counts

        # Run the diffusion process for the specified number of steps
        for step in range(time_horizon):
            # Calculate the total tokens to be passed from each node
            total_outgoing_tokens_2 = {node: 0 for node in G.nodes}
            for node in G.nodes:
                for neighbor in G.neighbors(node):
                    edge_strength = G.edges[node, neighbor]['strength']
                    edge_weight = strength_to_weight[edge_strength]
                    total_outgoing_tokens_2[node] += edge_weight

            # Calculate the new tokens at each node
            new_tokens_2 = tokens_2.copy()  # Start with the current tokens
            for node in G.nodes:
                for neighbor in G.neighbors(node):
                    edge_strength = G.edges[node, neighbor]['strength']
                    edge_weight = strength_to_weight[edge_strength]
                    tokens_passed = tokens_2[node] * (edge_weight / total_outgoing_tokens_2[node])
                    new_tokens_2[neighbor] += tokens_passed
                    if node not in intervention_nodes_2:  # Only subtract tokens if the node is not an intervention node
                        new_tokens_2[node] -= tokens_passed

            # Update the tokens
            tokens_2 = new_tokens_2

            # Inject tokens at the specified nodes at a fixed rate
            for node, rate in token_dict_2.items():
                tokens_2[node] += rate

            # Store the token counts at this time step
            for node in G.nodes:
                token_counts_over_time_2[G.nodes[node]['label']].append(tokens_2[node])

        # Convert the token counts to a DataFrame and transpose it
        df_token_counts_2 = pd.DataFrame(token_counts_over_time_2).T

        # Calculate the percentage of nodes with non-zero token counts
        non_zero_tokens_percentage_2 = (df_token_counts_2.astype(bool).sum(axis=1) > 0).sum() / len(df_token_counts_1) * 100


    col1,col2 = st.columns(2)

    ### % System leverage ###

    with col1:

        st.markdown("#### % System Controlability: Intervention Package 1")

        # Create a gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=non_zero_tokens_percentage_1.mean(),
            number={'suffix': "%"},  # Add a '%' suffix to the number displayed
            gauge={'axis': {'range': [None, 100]}}
        ))

        # Display the gauge chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.markdown("#### % System Controlability: Intervention Package 2")

        # Create a gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=non_zero_tokens_percentage_2.mean(),
            number={'suffix': "%"},  # Add a '%' suffix to the number displayed
            gauge={'axis': {'range': [None, 100]}}
        ))

        # Display the gauge chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    ### Leverage on outcome nodes ###

    with col1:

        st.markdown("#### Leverage (strength of causal effects) on outcome nodes: Intervention Package 1")

        # Create a line plot for the outcome nodes
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for outcome_node in outcome_nodes_1:
            # Compute a moving average with a window size of 5
            smoothed_token_counts_1 = df_token_counts_1.loc[outcome_node].rolling(window=rolling_window).mean()
            ax.plot(smoothed_token_counts_1, label=outcome_node)
        
        plt.xlabel('Time step', fontsize=20)
        plt.ylabel('Token count (~strength of causal effect)', fontsize=20)
        plt.legend(prop={'size': 20})
        plt.grid(True)

        # Save and read graph as png file (on cloud)
        try:
            path = './streamlit/static'
            fig.savefig(f'{path}/pulse_causal_effects_line_compare_1.png')      
            image = Image.open(f'{path}/pulse_causal_effects_line_compare_1.png')
            st.image(image)    
            
        # Save and read graph as HTML file (locally)
        except:
            path = 'static'
            fig.savefig(f'{path}/pulse_causal_effects_line_compare_1.png')      
            image = Image.open(f'{path}/pulse_causal_effects_line_compare_1.png')
            st.image(image)

    with col2:

        st.markdown("#### Leverage (strength of causal effects) on outcome nodes: Intervention Package 2")

        # Create a line plot for the outcome nodes
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for outcome_node in outcome_nodes_2:
            # Compute a moving average with a window size of 5
            smoothed_token_counts_2 = df_token_counts_2.loc[outcome_node].rolling(window=rolling_window).mean()
            ax.plot(smoothed_token_counts_2, label=outcome_node)
        
        plt.xlabel('Time step', fontsize=20)
        plt.ylabel('Token count (~strength of causal effect)', fontsize=20)
        plt.legend(prop={'size': 20})
        plt.grid(True)

        # Save and read graph as png file (on cloud)
        try:
            path = './streamlit/static'
            fig.savefig(f'{path}/pulse_causal_effects_line_compare_2.png')      
            image = Image.open(f'{path}/pulse_causal_effects_line_compare_2.png')
            st.image(image)    
            
        # Save and read graph as HTML file (locally)
        except:
            path = 'static'
            fig.savefig(f'{path}/pulse_causal_effects_line_compare_2.png')      
            image = Image.open(f'{path}/pulse_causal_effects_line_compare_2.png')
            st.image(image)

    with col1:

        # Create a heatmap of the token counts
        st.markdown("#### Leverage (strength of causal effects) on all nodes (red = outcome nodes; blue = intervention nodes): Intervention Pakage 1")

        # Apply log scale if log_scale is True
        if log_scale:
            df_token_counts_1 = np.log1p(df_token_counts_1)

        if st.session_state.fishery == 'Jumbo Flying Squid':
            fig, ax = plt.subplots(figsize=(12, 16))
        else:
            fig, ax = plt.subplots(figsize=(12, 12))
        
        sns.heatmap(df_token_counts_1, annot=True, fmt=".1f", cmap='magma', annot_kws={"size": 7}, cbar=False, ax=ax)

        # Highlight outcome nodes in red
        for label in ax.get_yticklabels():
            if label.get_text() in outcome_nodes_1:
                label.set_color('red')

        # Highlight nodes that have non-zero token counts (TOKENS column in df dataframe) in blue
        for label in ax.get_yticklabels():
            if label.get_text() in intervention_nodes_1:
                label.set_color('blue')

        plt.xlabel('Time step')
        plt.tight_layout()

        # Save and read graph as png file (on cloud)
        try:
            path = './streamlit/static'
            fig.savefig(f'{path}/pulse_causal_effects_heat_compare_1.png')      
            image = Image.open(f'{path}/pulse_causal_effects_heat_compare_1.png')
            left_co, right_co = st.columns([0.05,0.95])
            with right_co:
                st.image(image, use_column_width='auto')    
            
        # Save and read graph as HTML file (locally)
        except:
            path = 'static'
            fig.savefig(f'{path}/pulse_causal_effects_heat_compare_1.png')      
            image = Image.open(f'{path}/pulse_causal_effects_heat_compare_1.png')
            left_co, right_co = st.columns([0.05,0.95])
            with right_co:
                st.image(image, use_column_width='auto')           

    with col2:

        # Create a heatmap of the token counts
        st.markdown("#### Leverage (strength of causal effects) on all nodes (red = outcome nodes; blue = intervention nodes): Intervention Pakage 2")

        # Apply log scale if log_scale is True
        if log_scale:
            df_token_counts_2 = np.log1p(df_token_counts_2)

        if st.session_state.fishery == 'Jumbo Flying Squid':
            fig, ax = plt.subplots(figsize=(12, 16))
        else:
            fig, ax = plt.subplots(figsize=(12, 12))
            
        sns.heatmap(df_token_counts_2, annot=True, fmt=".1f", cmap='magma', annot_kws={"size": 7}, cbar=False, ax=ax)

        # Highlight outcome nodes in red
        for label in ax.get_yticklabels():
            if label.get_text() in outcome_nodes_2:
                label.set_color('red')

        # Highlight nodes that have non-zero token counts (TOKENS column in df dataframe) in blue
        for label in ax.get_yticklabels():
            if label.get_text() in intervention_nodes_2:
                label.set_color('blue')

        plt.xlabel('Time step')
        plt.tight_layout()

        # Save and read graph as png file (on cloud)
        try:
            path = './streamlit/static'
            fig.savefig(f'{path}/pulse_causal_effects_heat_compare_2.png')      
            image = Image.open(f'{path}/pulse_causal_effects_heat_compare_2.png')
            left_co, right_co = st.columns([0.05,0.95])
            with right_co:
                st.image(image, use_column_width='auto')    
            
        # Save and read graph as HTML file (locally)
        except:
            path = 'static'
            fig.savefig(f'{path}/pulse_causal_effects_heat_compare_2.png')      
            image = Image.open(f'{path}/pulse_causal_effects_heat_compare_2.png')
            left_co, right_co = st.columns([0.05,0.95])
            with right_co:
                st.image(image, use_column_width='auto') 


    with col1:

        st.markdown("#### Leverage (cumulative strength of causal effects = total area under token count curves) on all nodes: Intervention Package 1")

        if st.session_state.fishery == 'Jumbo Flying Squid':
            fig, ax = plt.subplots(figsize=(15, 16))
        else:
            fig, ax = plt.subplots(figsize=(15, 10))

        # Plot a bar chart showing the total area under the df_token_counts curve for all nodes
        for index, value in enumerate(df_token_counts_1.sum(axis=1)):
            sorted_sums = df_token_counts_1.sum(axis=1).sort_values()
            sorted_indices = sorted_sums.index
            colors = ['red' if idx in outcome_nodes_1 else 'blue' if idx in intervention_nodes_1 else 'gray' for idx in sorted_indices]
            ax.barh(range(len(sorted_indices)), sorted_sums, color=colors, alpha=0.5)
            ax.set_yticks(range(len(sorted_indices)))
            ax.set_yticklabels(sorted_indices)
            for label in ax.get_yticklabels():
                label.set_color('red' if label.get_text() in outcome_nodes_1 else 'blue' if label.get_text() in intervention_nodes_1 else 'gray')
            ax.set_xlabel('Total Area Under Token Count Curve (~cumulative strength of causal effect)')
        plt.tight_layout()
        
        # Save and read graph as png file (on cloud)
        try:
            path = './streamlit/static'
            fig.savefig(f'{path}/pulse_causal_effects_bar.png')
            image = Image.open(f'{path}/pulse_causal_effects_bar.png')
            st.image(image, use_column_width="auto")

        # Save and read graph as HTML file (locally)
        except:
            path = 'static'
            fig.savefig(f'{path}/pulse_causal_effects_bar.png')
            image = Image.open(f'{path}/pulse_causal_effects_bar.png')
            st.image(image, use_column_width="auto")      

    with col2:

        st.markdown("#### Leverage (cumulative strength of causal effects = total area under token count curves) on all nodes: Intervention Package 2")

        if st.session_state.fishery == 'Jumbo Flying Squid':
            fig, ax = plt.subplots(figsize=(15, 16))
        else:
            fig, ax = plt.subplots(figsize=(15, 10))

        # Plot a bar chart showing the total area under the df_token_counts curve for all nodes
        for index, value in enumerate(df_token_counts_2.sum(axis=1)):
            sorted_sums = df_token_counts_2.sum(axis=1).sort_values()
            sorted_indices = sorted_sums.index
            colors = ['red' if idx in outcome_nodes_2 else 'blue' if idx in intervention_nodes_2 else 'gray' for idx in sorted_indices]
            ax.barh(range(len(sorted_indices)), sorted_sums, color=colors, alpha=0.5)
            ax.set_yticks(range(len(sorted_indices)))
            ax.set_yticklabels(sorted_indices)
            for label in ax.get_yticklabels():
                label.set_color('red' if label.get_text() in outcome_nodes_2 else 'blue' if label.get_text() in intervention_nodes_2 else 'gray')
            ax.set_xlabel('Total Area Under Token Count Curve (~cumulative strength of causal effect)')
        plt.tight_layout()
        
        # Save and read graph as png file (on cloud)
        try:
            path = './streamlit/static'
            fig.savefig(f'{path}/pulse_causal_effects_bar.png')
            image = Image.open(f'{path}/pulse_causal_effects_bar.png')
            st.image(image, use_column_width="auto")

        # Save and read graph as HTML file (locally)
        except:
            path = 'static'
            fig.savefig(f'{path}/pulse_causal_effects_bar.png')
            image = Image.open(f'{path}/pulse_causal_effects_bar.png')
            st.image(image, use_column_width="auto")   


    # Get the dataframes of token counts for each intervention package
    df1 = df_token_counts_1
    df2 = df_token_counts_2

    # Determine the number of rows for the subplots
    num_rows = len(df1) // 2 + len(df1) % 2

    # Determine max value from df1 and df2 (the max value across both dataframes)
    max_value = max(df1.values.max(), df2.values.max())
    
    # Create a new figure with subplots
    fig, axs = plt.subplots(num_rows, 2, figsize=(16, 3*num_rows))

    # Flatten the axes array
    axs = axs.flatten()

    # Get the main factors dataframe from the session state
    factors_df = st.session_state.df_factors

    # Iterate over each row in the DataFrame
    for ax, (index1, row1), (index2, row2) in zip(axs, df1.iterrows(), df2.iterrows()):
        ax.plot(row1.rolling(window=rolling_window).mean(), label='Intervention Package 1')
        ax.plot(row2.rolling(window=rolling_window).mean(), label='Intervention Package 2')
        total_area_1 = row1.sum()
        total_area_2 = row2.sum()
        ax.text(0.17, 0.98, f'Cumulative Strength of Effect (IP1): {total_area_1:.1f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.17, 0.90, f'Cumulative Strength of Effect (IP2): {total_area_2:.1f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        # Set the title to be the row name
        if index1 in outcome_nodes_1:
            ax.set_title(index1, fontsize=20, color='red')  # Increase plot title font and set color to red for outcome nodes
        if index2 in outcome_nodes_2:
            ax.set_title(index2, fontsize=20, color='red')  # Increase plot title font and set color to red for outcome nodes
        if index1 in intervention_nodes_1:
            ax.set_title(index1, fontsize=20, color='blue')
            # Get the number of tokens for the current node
            num_tokens = edited_df_1[edited_df_1['long_name'] == index1]['TOKENS'].values[0]
            # Add the number of tokens as a text box in the subplot
            ax.text(0.35, 0.9, f'IP1 Tokens: {num_tokens}', transform=ax.transAxes, fontsize=24, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))        
        if index2 in intervention_nodes_2:
            ax.set_title(index2, fontsize=20, color='blue')
            # Get the number of tokens for the current node
            num_tokens = edited_df_2[edited_df_2['long_name'] == index2]['TOKENS'].values[0]
            # Add the number of tokens as a text box in the subplot
            ax.text(0.35, 0.9, f'IP2 Tokens: {num_tokens}', transform=ax.transAxes, fontsize=24, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))          
        else:
            ax.set_title(index1, fontsize=20)  # Increase plot title font
        ax.set_xlabel('Time step')
        ax.set_ylabel('Token count\n(~strength of causal effect)')
        ax.set_ylim([0, max_value])  # Set the y-axis limits
        ax.legend()  # Add legend

    # Remove unused subplots
    for ax in axs[max(len(df1), len(df2)):]:
        fig.delaxes(ax)

    # Adjust the layout
    plt.tight_layout()

    # Save and read graph as png file (on cloud)
    try:
        path = './streamlit/static'
        fig.savefig(f'{path}/pulse_causal_effects_lines_compare_1and2.png')      
        image = Image.open(f'{path}/pulse_causal_effects_lines_compare_1and2.png')
        st.image(image, use_column_width=True)  
        
    # Save and read graph as HTML file (locally)
    except:
        path = 'static'
        fig.savefig(f'{path}/pulse_causal_effects_lines_compare_1and2.png')      
        image = Image.open(f'{path}/pulse_causal_effects_lines_compare_1and2.png')
        st.image(image, use_column_width=True)



### DEAP OPTIMIZATION ###

def pulse_diffusion_network_model_DEAP(G, initial_tokens, num_steps, log_scale=False):

    # Define the strength to weight mapping
    strength_to_weight = {'strong': 3, 'medium': 2, 'weak': 1}

    # Initialize the tokens at each node to zero
    tokens = {node: 0 for node in G.nodes}

    # Update the tokens for the nodes specified in the initial_tokens dictionary
    for node, token in initial_tokens.items():
        tokens[node] = token

    # Initialize a dictionary to store the token counts at each time step
    # Use the label attribute as the key
    token_counts_over_time = {G.nodes[node]['label']: [tokens[node]] for node in G.nodes}  # Include the initial token counts

    # Run the diffusion process for the specified number of steps
    for step in range(num_steps):
        # Calculate the total tokens to be passed from each node
        total_outgoing_tokens = {node: 0 for node in G.nodes}
        for node in G.nodes:
            for neighbor in G.neighbors(node):
                edge_strength = G.edges[node, neighbor]['strength']
                edge_weight = strength_to_weight[edge_strength]
                total_outgoing_tokens[node] += edge_weight

        # Calculate the new tokens at each node
        new_tokens = tokens.copy()  # Start with the current tokens
        for node in G.nodes:
            for neighbor in G.neighbors(node):
                edge_strength = G.edges[node, neighbor]['strength']
                edge_weight = strength_to_weight[edge_strength]
                tokens_passed = tokens[node] * (edge_weight / total_outgoing_tokens[node])
                if np.isfinite(tokens_passed):
                    new_tokens[neighbor] += tokens_passed
                    new_tokens[node] -= tokens_passed  # Redistribute the passed tokens back to the node

        # Update the tokens
        tokens = new_tokens

        # Store the token counts at this time step
        for node in G.nodes:
            token_counts_over_time[G.nodes[node]['label']].append(tokens[node])

    # Convert the token counts to a DataFrame and transpose it
    df_token_counts = pd.DataFrame(token_counts_over_time).T

    # Calculate the percentage of nodes with non-zero token counts
    non_zero_tokens_percentage = (df_token_counts.astype(bool).sum(axis=1) > 0).sum() / len(df_token_counts) * 100

    # Apply log scale if log_scale is True
    if log_scale:
        df_token_counts = np.log1p(df_token_counts)

    return df_token_counts

toolbox = base.Toolbox()

# Define the fitness function
def evaluate(individual):
    # Extract the nodes and token allocations from the individual
    nodes = individual[:3]
    token_allocations = individual[3:]

    # Ensure the token allocations sum to 100
    if sum(token_allocations) != 100:
        return -np.inf, -np.inf, -np.inf

    # Create the initial tokens dictionary
    initial_tokens = {node: allocation for node, allocation in zip(nodes, token_allocations)}

    G_ALL = plot_relationships('All relationships',True,'no_display')

    # Run the pulse diffusion network model
    df_token_counts = pulse_diffusion_network_model_DEAP(G_ALL, initial_tokens, 50, log_scale=False)

    # Calculate the non_zero_tokens_percentage
    non_zero_tokens_percentage = (df_token_counts.astype(bool).sum(axis=1) > 0).sum() / len(df_token_counts) * 100

    # Calculate the sum of the area under the token count timeseries for the outcome nodes
    factors_df = st.session_state.df_factors
    factors_df['OUTCOME NODE'] = factors_df['domain_name'].apply(lambda x: True if x == 'FOCAL FACTORS' else False)
    selected_outcome_nodes =  st.session_state.selected_outcome_nodes
    factors_df['OUTCOME NODE'] = factors_df['long_name'].apply(lambda x: True if x in selected_outcome_nodes else False)
    outcome_nodes = factors_df[factors_df['OUTCOME NODE']]['long_name'].tolist()
    
    # Calculate the area under the curve for all timesteps
    area_under_curve = np.trapz(df_token_counts.loc[outcome_nodes].values, axis=1).sum()
    # Calculate the area under the curve for the first 10 timesteps only
    area_under_curve_first_10 = np.trapz(df_token_counts.loc[outcome_nodes].values[:,:10], axis=1).sum()

    # Calculate the viability objective
    controllability_mapping = {'low': 1, 'medium': 2, 'high': 4, 'uncontrollable':0}
    knowledge_mapping = {'low': 1, 'medium': 2, 'high': 4, 'uncontrollable':0}
    predictability_mapping = {'low': 1, 'medium': 2, 'high': 3, 'uncontrollable':0}
    measurability_cost_mapping = {'low': 3, 'medium': 2, 'high': 1}
    opportunity_mapping = {'Active': 1, 'Potential': 2, 'No': 4}
    score = 0
    for node in nodes:
        node_data = factors_df.loc[node]
        score += controllability_mapping[node_data['controllability']]
        score += knowledge_mapping[node_data['level of knowledge']]
        score += predictability_mapping[node_data['predictability']]
        score += measurability_cost_mapping[node_data['measurability cost']]
        score += opportunity_mapping[node_data['interventions']]

    return non_zero_tokens_percentage, area_under_curve, score, area_under_curve_first_10

# Define the individual
def create_individual():
    factors_df = st.session_state.df_factors
    nodes = random.sample(factors_df[factors_df['intervenable'] == 'yes'].index.tolist(), 3)
    token_allocations = [toolbox.token_attr() for _ in range(2)]
    token_allocations.append(100 - sum(token_allocations))
    return creator.Individual(nodes + token_allocations)

def custom_mutate(individual, indpb):
    factors_df = st.session_state.df_factors
    # Mutate the nodes
    for i in range(3):
        if random.random() < indpb:
            available_nodes = [n for n in factors_df[factors_df['intervenable'] == 'yes'].index.tolist() if n not in individual[:3]]
            if available_nodes:  # only mutate if there are available nodes
                individual[i] = random.choice(available_nodes)

    # Mutate the token allocations
    for i in range(3, 5):
        if random.random() < indpb:
            individual[i] = random.randint(0, 50)

    # Ensure the token allocations sum to 100
    individual[5] = 100 - sum(individual[3:5])

    return individual,

def custom_crossover(ind1, ind2):
    # Perform two-point crossover on the nodes
    tools.cxTwoPoint(ind1[:3], ind2[:3])

    # Perform two-point crossover on the first two token allocations
    tools.cxTwoPoint(ind1[3:5], ind2[3:5])

    # Ensure the token allocations sum to 100
    ind1[5] = 100 - sum(ind1[3:5])
    ind2[5] = 100 - sum(ind2[3:5])

    return ind1, ind2

# Color code the dataframe using pandas styler object

def color_lowmedhigh(val):
    color = 'lightgreen' if val == 'low' else 'lightyellow' if val == 'medium' else 'lightcoral'
    return 'background-color: %s' % color

def color_intervenable(val):
    color = 'lightgreen' if val == 'yes' else 'red'
    return 'background-color: %s' % color

def color_interventions(val):
    if val == 'None':
        color = 'lightblue'
    elif val == 'Active':
        color = 'lightgreen'
    else:
        color = 'lightcoral'
    return 'background-color: %s' % color


### VARS SA ###


def model_response_sum_outcome_nodes(G, initial_tokens, num_steps, log_scale=False):
    # Run the pulse diffusion network model
    df_token_counts = pulse_diffusion_network_model_DEAP(G, initial_tokens, num_steps, log_scale=log_scale)

    # Get the outcome nodes from the global session state if available
    if 'df_factors' in st.session_state:
        factors_df = st.session_state.df_factors
        factors_df['OUTCOME NODE'] = factors_df['domain_name'].apply(lambda x: True if x == 'FOCAL FACTORS' else False)
        outcome_nodes = factors_df[factors_df['OUTCOME NODE']]['long_name'].tolist()
    else:
        # If the session state does not contain the factors dataframe, define outcome nodes manually
        # This is a placeholder and should be replaced with the actual method of obtaining outcome nodes
        outcome_nodes = ['Node1', 'Node2']  # Placeholder: Replace with actual outcome nodes

    # Calculate the sum of token counts for the outcome nodes
    sum_token_counts_outcome_nodes = df_token_counts.loc[outcome_nodes].sum().sum()

    return sum_token_counts_outcome_nodes
