from utilities import *

load_factors(st.session_state.sheet_id)
load_relationships(st.session_state.sheet_id)
load_domains(st.session_state.sheet_id)
G=plot_relationships('All relationships',True,'no_display')

factors_df = st.session_state.df_factors
factors_df = factors_df.drop(['short_name', 'predictability', 'measurability cost'], axis=1)
factors_df['OUTCOME NODE'] = factors_df['domain_name'].apply(lambda x: True if x == 'FOCAL FACTORS' else False)
factors_df['TOKENS'] = 0

st.title(st.session_state.fishery)

with st.expander('Influence Diagram'):

    st.title('Influence Diagram')

    relationships_filter = st.selectbox('Select relationships:', ['All relationships','Strong only'])
    domains_to_keep = st.multiselect('Select domains:', factors_df['domain_name'].unique().tolist(), default=factors_df['domain_name'].unique().tolist())
    domains_to_remove = factors_df[~factors_df['domain_name'].isin(domains_to_keep)]['domain_id'].tolist()
    intervenable_filter = st.selectbox('Select intervenable factors:', ['All factors','Intervenable Only'])

    G=plot_relationships_interventionlab(relationships_filter,True,'display', domains_to_remove, intervenable_filter)


with st.expander('Exploratory Scenario Analysis (single intervention package)'):
    
    st.title('Intervention Package')

    factors_df_display = factors_df[['factor_id',"OUTCOME NODE", "long_name", "TOKENS", "interventions", "intervenable", "domain_name"]]
    edited_df = st.data_editor(factors_df_display, use_container_width=False, height=1750, key='')
    token_dict = edited_df[edited_df['TOKENS'] != 0].set_index('factor_id')['TOKENS'].to_dict()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col4:
        log_scale = st.checkbox('Use log scale?', key='log_scale_1')
    with col1:
        diffusion_model = st.selectbox('Choose a model:', ('one-time investment', 'continuous investment'), key='diffusion_model_1')
    with col2:
        time_horizon = st.slider('Time horizon (timesteps):', 1, 50, 25, key='time_horizon_1')
    with col3: 
        rolling_window = st.slider('Rolling window (timesteps):', 1, 5, 1, key='rolling_window_1')

    if st.button('Run simulation', key='run_simulation_1'):

        with st.spinner('Running analysis...'):

            if diffusion_model == 'one-time investment':
                pulse_diffusion_network_model(G, token_dict, time_horizon, edited_df, log_scale, rolling_window, vertical=False, case='')
            else:
                flow_diffusion_network_model(G, token_dict, time_horizon, edited_df, log_scale, rolling_window)


with st.expander('Exploratory Scenario Analysis (compare two intervention packages)'):

    with st.form("my_form"):

        col1, col2 = st.columns(2)

        with col1:

            st.title('Intervention Package 1')

            factors_df_display_1 = factors_df[['factor_id',"OUTCOME NODE", "long_name", "TOKENS", "interventions", "intervenable", "domain_name"]]
            edited_df_1 = st.data_editor(factors_df_display_1, use_container_width=False, height=1750, key='case1')
            token_dict_1 = edited_df_1[edited_df_1['TOKENS'] != 0].set_index('factor_id')['TOKENS'].to_dict()

        with col2:

            st.title('Intervention Package 2')

            factors_df_display_2 = factors_df[['factor_id',"OUTCOME NODE", "long_name", "TOKENS", "interventions", "intervenable", "domain_name"]]
            edited_df_2 = st.data_editor(factors_df_display_2, use_container_width=False, height=1750, key='case2')
            token_dict_2 = edited_df_2[edited_df_2['TOKENS'] != 0].set_index('factor_id')['TOKENS'].to_dict()

        
        col1, col2, col3, col4 = st.columns(4)
    
        with col4:
            log_scale = st.checkbox('Use log scale?', key='log_scale_compare')
        with col1:
            diffusion_model = st.selectbox('Choose a model:', ('one-time investment', 'continuous investment'), key='diffusion_model_compare')
        with col2:
            time_horizon = st.slider('Time horizon (timesteps):', 1, 50, 25, key='time_horizon_compare')
        with col3: 
            rolling_window = st.slider('Rolling window (timesteps):', 1, 5, 1, key='rolling_window_compare')


        # Every form must have a submit button.
        submitted = st.form_submit_button("Run comparisons")
        if submitted:
            with st.spinner('Running comparisons...'):
                diffusion_model_compare(G, token_dict_1, token_dict_2, diffusion_model, time_horizon, edited_df_1, edited_df_2, log_scale, rolling_window)

            


    # col1, col2 = st.columns(2)

    # with col1:

    #     st.title('Case 1')

    #     factors_df_display = factors_df[['factor_id',"OUTCOME NODE", "long_name", "TOKENS", "interventions", "intervenable", "domain_name"]]
    #     edited_df = st.data_editor(factors_df_display, use_container_width=False, height=1750, key='case2')
    #     token_dict = edited_df[edited_df['TOKENS'] != 0].set_index('factor_id')['TOKENS'].to_dict()
        
    #     col3, col4, col5, col6 = st.columns(4)
        
    #     with col3:
    #         log_scale = st.checkbox('Use log scale?', key='log_scale_2')
    #     with col4:
    #         diffusion_model = st.selectbox('Choose a model:', ('one-time investment', 'continuous investment'), key='diffusion_model_2')
    #     with col5:
    #         time_horizon = st.slider('Time horizon (timesteps):', 1, 50, 25, key='time_horizon_2')
    #     with col6: 
    #         rolling_window = st.slider('Rolling window (timesteps):', 1, 5, 1, key='rolling_window_2')

    #     if st.button('Run simulation', key='run_simulation_2'):

    #         if diffusion_model == 'one-time investment':
    #             pulse_diffusion_network_model(G, token_dict, time_horizon, edited_df, log_scale, rolling_window, vertical=True, case='case1')
    #         else:
    #             flow_diffusion_network_model(G, token_dict, time_horizon, edited_df, log_scale, rolling_window)


    # with col2:
        
    #     st.title('Case 2')

    #     factors_df_display = factors_df[['factor_id',"OUTCOME NODE", "long_name", "TOKENS", "interventions", "intervenable", "domain_name"]]
    #     edited_df = st.data_editor(factors_df_display, use_container_width=False, height=1750, key='case3')
    #     token_dict = edited_df[edited_df['TOKENS'] != 0].set_index('factor_id')['TOKENS'].to_dict()
        
    #     col7, col8, col9, col10 = st.columns(4)
        
    #     with col7:
    #         log_scale = st.checkbox('Use log scale?', key='log_scale_3')
    #     with col8:
    #         diffusion_model = st.selectbox('Choose a model:', ('one-time investment', 'continuous investment'), key='diffusion_model_3')
    #     with col9:
    #         time_horizon = st.slider('Time horizon (timesteps):', 1, 50, 25, key='time_horizon_3')
    #     with col10: 
    #         rolling_window = st.slider('Rolling window (timesteps):', 1, 5, 1, key='rolling_window_3')

    #     if st.button('Run simulation', key='run_simulation_3'):

    #         if diffusion_model == 'one-time investment':
    #             pulse_diffusion_network_model(G, token_dict, time_horizon, edited_df, log_scale, rolling_window, vertical=True, case='case2')
    #         else:
    #             flow_diffusion_network_model(G, token_dict, time_horizon, edited_df, log_scale, rolling_window)

    # if st.button('Plot Comparisons', key='plot_comparison'):
    
    #     # Get the dataframes from the session state
    #     df1 = st.session_state.df_token_counts_1
    #     df2 = st.session_state.df_token_counts_2

    #     # Determine the number of rows for the subplots
    #     num_rows = max(len(df1), len(df2))

    #     # Create a new figure with subplots
    #     fig, axs = plt.subplots(num_rows, 2, figsize=(16, 3*num_rows), dpi=100)

    #     # Flatten the axes array
    #     axs = axs.flatten()

    #     # Get the outcome nodes
    #     outcome_nodes = factors_df[factors_df['OUTCOME NODE'] == True]['long_name'].tolist()

    #     # Iterate over each row in the DataFrame
    #     for ax, (index1, row1), (index2, row2) in zip(axs, df1.iterrows(), df2.iterrows()):
    #         ax.plot(row1, label='Case 1')
    #         ax.plot(row2, label='Case 2')
    #         # Set the title to be the row name
    #         if index1 in outcome_nodes:
    #             ax.set_title(index1, fontsize=20, color='red')  # Increase plot title font and set color to red for outcome nodes
    #         else:
    #             ax.set_title(index1, fontsize=20)  # Increase plot title font
    #         ax.set_xlabel('Time step')
    #         ax.set_ylabel('Token count')
    #         ax.legend()  # Add legend

    #     # Remove unused subplots
    #     for ax in axs[max(len(df1), len(df2)):]:
    #         fig.delaxes(ax)

    #     # Adjust the layout
    #     plt.tight_layout()

    #     # Display the plot in Streamlit
    #     st.pyplot(fig)


with st.expander('Optimisation Analysis'):

    st.title('Optimisation Analysis')

    outcome_node_options = factors_df[factors_df['OUTCOME NODE'] == True]['long_name'].tolist()
    selected_outcome_nodes = st.multiselect('Select outcome nodes (these will be used to quantify the "Effect" objective of potential interventions):', factors_df['long_name'].tolist(), default=outcome_node_options)
    factors_df['OUTCOME NODE'] = factors_df['long_name'].apply(lambda x: True if x in selected_outcome_nodes else False)

    options = st.multiselect(
    'What objectives would you like to optimise?',
    ['Control', 'Effect', 'Effect (short term)', 'Viability'],
    ['Control', 'Effect', 'Effect (short term)', 'Viability'])

    if st.button('Run optimisation', key='run_optimisation'):

        with st.spinner('Please wait...'):

            # Define the individual and population
            
            if 'Control' in options and 'Effect' in options and 'Viability' in options and 'Effect (short term)' in options:
                creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0))
            elif 'Control' in options and 'Effect' in options and 'Viability' not in options and 'Effect (short term)' in options:
                creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, -np.inf, 1.0))
            elif 'Control' in options and 'Effect' not in options and 'Viability' in options and 'Effect (short term)' in options:
                creator.create("FitnessMax", base.Fitness, weights=(1.0, -np.inf, 1.0, 1.0))
            elif 'Control' not in options and 'Effect' in options and 'Viability' in options and 'Effect (short term)' in options:
                creator.create("FitnessMax", base.Fitness, weights=(-np.inf, 1.0, 1.0, 1.0))
            elif 'Control' in options and 'Effect' not in options and 'Viability' not in options and 'Effect (short term)' in options:
                creator.create("FitnessMax", base.Fitness, weights=(1.0, -np.inf, -np.inf, 1.0))
            elif 'Control' not in options and 'Effect' in options and 'Viability' not in options and 'Effect (short term)' in options:
                creator.create("FitnessMax", base.Fitness, weights=(-np.inf, 1.0, -np.inf, 1.0))
            elif 'Control' not in options and 'Effect' not in options and 'Viability' in options and 'Effect (short term)' in options:
                creator.create("FitnessMax", base.Fitness, weights=(-np.inf, -np.inf, 1.0, 1.0))
            elif 'Control' in options and 'Effect' in options and 'Viability' in options and 'Effect (short term)' not in options:
                creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0, -np.inf))
            elif 'Control' in options and 'Effect' not in options and 'Viability' in options and 'Effect (short term)' not in options:
                creator.create("FitnessMax", base.Fitness, weights=(1.0, -np.inf, 1.0, -np.inf))
            elif 'Control' not in options and 'Effect' in options and 'Viability' in options and 'Effect (short term)' not in options:
                creator.create("FitnessMax", base.Fitness, weights=(-np.inf, 1.0, 1.0, -np.inf))
            elif 'Control' in options and 'Effect' not in options and 'Viability' not in options and 'Effect (short term)' not in options:
                creator.create("FitnessMax", base.Fitness, weights=(1.0, -np.inf, -np.inf, -np.inf))
            elif 'Control' not in options and 'Effect' in options and 'Viability' not in options and 'Effect (short term)' not in options:
                creator.create("FitnessMax", base.Fitness, weights=(-np.inf, 1.0, -np.inf, -np.inf))
            elif 'Control' not in options and 'Effect' not in options and 'Viability' in options and 'Effect (short term)' not in options:
                creator.create("FitnessMax", base.Fitness, weights=(-np.inf, -np.inf, 1.0, -np.inf))
            elif 'Control' not in options and 'Effect' not in options and 'Viability' not in options and 'Effect (short term)' in options:
                creator.create("FitnessMax", base.Fitness, weights=(-np.inf, -np.inf, -np.inf, 1.0))
            elif 'Control' in options and 'Effect' in options and 'Viability' not in options and 'Effect (short term)' not in options:
                creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, -np.inf, -np.inf))
            elif 'Control' not in options and 'Effect' not in options and 'Viability' not in options and 'Effect (short term)' not in options:
                creator.create("FitnessMax", base.Fitness, weights=(-np.inf, -np.inf, -np.inf, -np.inf))
            
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            #toolbox = base.Toolbox()

            # Define the attribute generators
            toolbox.register("node_attr", random.choice, factors_df[factors_df['intervenable'] == 'yes'].index.tolist())
            toolbox.register("token_attr", random.randint, 0, 50)

            toolbox.register("individual", create_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            # Register the genetic operators
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", custom_crossover)
            toolbox.register("mutate", custom_mutate, mu=0, sigma=1, indpb=0.1)
            toolbox.register("select", tools.selTournament, tournsize=6)

            # Run the genetic algorithm
            pop = toolbox.population(n=50)
            hof = tools.HallOfFame(10)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)

            pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.15, mutpb=0.1, ngen=300, stats=stats, halloffame=hof, verbose=True)
            st.success('Done!') 
        
        st.balloons()

        col1,col2 = st.columns(2)

        with col1:

            st.subheader('Genetic Algorithm Results')
            # Extract the statistics
            avg_fitness_values = log.select('avg')

            # Create the plot
            plt.figure(figsize=(10, 5))
            plt.plot(avg_fitness_values, label=['Control','Effect', 'Viability', 'Effect (short term)'])
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Fitness Evolution')
            plt.legend()
            plt.grid(True)
            
            # Display the plot in Streamlit
            st.pyplot(plt)
        
        with col2:

            st.subheader('Hall of Fame: 10 best solutions')

            # Create a list of solutions
            solutions = [{"Intervention Nodes": str(ind[:3]), 
                        "Effort": str(ind[3:]), 
                        "Control": str(round(ind.fitness.values[0], 1)), 
                        "Effect": str(round(ind.fitness.values[1], 1)),
                        "Viability": str(round(ind.fitness.values[2], 1)),
                        "Effect (short term)": str(round(ind.fitness.values[3], 1))
                        } for ind in hof]

            # Display the solutions in a table
            st.table(pd.DataFrame(solutions))

            # Create a DataFrame for node names
            node_numbers = pd.DataFrame(solutions)['Intervention Nodes'].apply(lambda x: x.strip('[]').split(', ')).explode().unique()
            node_names = [factors_df.loc[int(i), 'long_name'] for i in node_numbers]

            # Create a lookup table
            lookup_table = pd.DataFrame({'Node Number': node_numbers, 'Node Name': node_names})

        # Join lookup_table and factors_df
        lookup_table['Node Number'] = lookup_table['Node Number'].astype('int64')
        merged_df = pd.merge(lookup_table, st.session_state.df_factors, left_on='Node Number', right_on='factor_id')

        # Select the required columns
        selected_columns = merged_df[['Node Number', 'Node Name', 'controllability', 'level of knowledge', 'predictability', 'measurability cost', 'intervenable', 'interventions']]

        styled_df = selected_columns.style.applymap(color_lowmedhigh, subset=['controllability','level of knowledge', 'predictability', 'measurability cost']).\
            applymap(color_intervenable, subset=['intervenable']).\
                applymap(color_interventions, subset=['interventions'])

        # Display the styled dataframe in Streamlit
        st.dataframe(styled_df)

# import openai

# openai.api_key = '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'

# def generate_response(prompt):
    
#     completion = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=prompt,
#         max_tokens=1024,
#         n=1,
#         stop=None,
#         temperature=0.5,)
    
#     message = completion.choices[0].text
#     return message.strip()

# st.title("Chatbot with OpenAI GPT-3")
# user_input = st.text_input("User Input")
# if user_input:
#     bot_response = generate_response(user_input)
#     st.write("Bot Response: ", bot_response)
