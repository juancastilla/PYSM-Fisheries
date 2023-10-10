from utilities import *

load_factors(st.session_state.sheet_id)
load_relationships(st.session_state.sheet_id)
load_domains(st.session_state.sheet_id)
G=plot_relationships('All relationships',True,'no_display')

factors_df = st.session_state.df_factors
factors_df = factors_df.drop(['domain_id', 'short_name', 'predictability', 'measurability cost'], axis=1)
factors_df['OUTCOME NODE'] = factors_df['domain_name'].apply(lambda x: True if x == 'FOCAL FACTORS' else False)
factors_df['TOKENS'] = 0

with st.expander('Force-directed plot'):

    st.title('Force-directed plot')

    G=plot_relationships('All relationships',True,'display')

col1, col2 = st.columns(2)

with col1:

    with st.expander('Exploratory Scenario Analysis (CASE 1)'):
        
        st.title('Exploratory Scenario Analysis')

        edited_df = st.data_editor(factors_df, use_container_width=False, height=1750, key='case1')
        token_dict = edited_df[edited_df['TOKENS'] != 0].set_index('factor_id')['TOKENS'].to_dict()
        log_scale = st.checkbox('Use log scale?', key='log_scale_1')

        if st.button('Run simulation', key='run_simulation_1'):
            pulse_diffusion_network_model(G, token_dict, 50, edited_df, log_scale)

with col2:

    with st.expander('Exploratory Scenario Analysis (CASE 2)'):
        
        st.title('Exploratory Scenario Analysis')

        edited_df = st.data_editor(factors_df, use_container_width=False, height=1750, key='case2')
        token_dict = edited_df[edited_df['TOKENS'] != 0].set_index('factor_id')['TOKENS'].to_dict()
        log_scale = st.checkbox('Use log scale?', key='log_scale_2')

        if st.button('Run simulation', key='run_simulation_2'):
            pulse_diffusion_network_model(G, token_dict, 50, edited_df, log_scale)

with st.expander('Optimisation Analysis'):
    st.title('Optimisation Analysis')
