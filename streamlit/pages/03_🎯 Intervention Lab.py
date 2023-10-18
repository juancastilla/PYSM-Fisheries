from utilities import *

load_factors(st.session_state.sheet_id)
load_relationships(st.session_state.sheet_id)
load_domains(st.session_state.sheet_id)
G=plot_relationships('All relationships',True,'no_display')

factors_df = st.session_state.df_factors
factors_df = factors_df.drop(['domain_id', 'short_name', 'predictability', 'measurability cost'], axis=1)
factors_df['OUTCOME NODE'] = factors_df['domain_name'].apply(lambda x: True if x == 'FOCAL FACTORS' else False)
factors_df['TOKENS'] = 0

st.title(st.session_state.fishery)

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
        diffusion_model = st.selectbox('Choose a diffusion model:', ('Pulse diffusion', 'Flow diffusion'), key='diffusion_model_1')

        if st.button('Run simulation', key='run_simulation_1'):

            if diffusion_model == 'Pulse diffusion':
                pulse_diffusion_network_model(G, token_dict, 50, edited_df, log_scale)
            else:
                flow_diffusion_network_model(G, token_dict, 50, edited_df, log_scale)

with col2:

    with st.expander('Exploratory Scenario Analysis (CASE 2)'):
        
        st.title('Exploratory Scenario Analysis')

        edited_df = st.data_editor(factors_df, use_container_width=False, height=1750, key='case2')
        token_dict = edited_df[edited_df['TOKENS'] != 0].set_index('factor_id')['TOKENS'].to_dict()
        log_scale = st.checkbox('Use log scale?', key='log_scale_2')
        diffusion_model = st.selectbox('Choose a diffusion model:', ('Pulse diffusion', 'Flow diffusion'), key='diffusion_model_2')

        if st.button('Run simulation', key='run_simulation_2'):

            if diffusion_model == 'Pulse diffusion':
                pulse_diffusion_network_model(G, token_dict, 50, edited_df, log_scale)
            else:
                flow_diffusion_network_model(G, token_dict, 50, edited_df, log_scale)

with st.expander('Optimisation Analysis'):

    st.title('Optimisation Analysis')

    options = st.multiselect(
    'What objectives do you want to optimise?',
    ['Controllability', 'Effect on Outcome Nodes'],
    ['Controllability', 'Effect on Outcome Nodes'])

    if st.button('Run optimisation', key='run_optimisation'):

        with st.spinner('Please wait...'):

            # Define the individual and population
            
            if 'Controllability' in options and 'Effect on Outcome Nodes' in options:
                creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
            elif 'Controllability' in options and 'Effect on Outcome Nodes' not in options:
                creator.create("FitnessMax", base.Fitness, weights=(1.0, 0.0))
            elif 'Effect on Outcome Nodes' in options and 'Controllability' not in options:
                creator.create("FitnessMax", base.Fitness, weights=(0.0, 1.0))
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
            if 'Controllability' in options and 'Effect on Outcome Nodes' in options:
                plt.plot(avg_fitness_values, label=['Controllability','Effect on Outcome Nodes'])
            elif 'Controllability' in options and 'Effect on Outcome Nodes' not in options:
                plt.plot(avg_fitness_values, label=['Controllability',''])
            elif 'Effect on Outcome Nodes' in options and 'Controllability' not in options:
                plt.plot(avg_fitness_values, label=['','Effect on Outcome Nodes'])

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
                        "Intervention Effort": str(ind[3:]), 
                        "Controllability": str(round(ind.fitness.values[0], 1)), 
                        "Effect on Outcome Nodes": str(round(ind.fitness.values[1], 1))} for ind in hof]

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
        selected_columns = merged_df[['Node Number', 'Node Name', 'controllability', 'level of knowledge', 'predictability', 'measurability cost', 'intervenable', 'Interventions']]

        styled_df = selected_columns.style.applymap(color_lowmedhigh, subset=['controllability','level of knowledge', 'predictability', 'measurability cost']).\
            applymap(color_intervenable, subset=['intervenable']).\
                applymap(color_interventions, subset=['Interventions'])

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
