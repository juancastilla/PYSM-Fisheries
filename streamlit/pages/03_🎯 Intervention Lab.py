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

        if sum(token_dict.values()) != 100:
            st.warning('The sum of tokens for the intervention package must equal 100 to run the simulation.')
        else:
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
            if sum(token_dict_1.values()) != 100 or sum(token_dict_2.values()) != 100:
                st.warning('The sum of tokens for each of the intervention packages must equal 100 to run the simulation.')
            else:
                with st.spinner('Running comparisons...'):
                    diffusion_model_compare(G, token_dict_1, token_dict_2, diffusion_model, time_horizon, edited_df_1, edited_df_2, log_scale, rolling_window)




with st.expander('Optimisation Analysis'):

    st.title('Optimisation Analysis')

    outcome_node_options = factors_df[factors_df['OUTCOME NODE'] == True]['long_name'].tolist()
    selected_outcome_nodes = st.multiselect('Select outcome nodes (these will be used to quantify the "Effect" objective of potential interventions):', factors_df['long_name'].tolist(), default=outcome_node_options)
    factors_df['OUTCOME NODE'] = factors_df['long_name'].apply(lambda x: True if x in selected_outcome_nodes else False)
    st.session_state.selected_outcome_nodes = selected_outcome_nodes

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
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(avg_fitness_values, label=['Control','Effect', 'Viability', 'Effect (short term)'])
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.set_title('Fitness Evolution')
            ax.legend()
            ax.grid(True)
            
            # Save and read graph as png file (on cloud)
            try:
                path = './streamlit/static'
                fig.savefig(f'{path}/optimisation_fitness.png')      
                image = Image.open(f'{path}/optimisation_fitness.png')
                st.image(image, use_column_width=True)  
                
            # Save and read graph as HTML file (locally)
            except:
                path = 'static'
                fig.savefig(f'{path}/optimisation_fitness.png')      
                image = Image.open(f'{path}/optimisation_fitness.png')
                st.image(image, use_column_width=True)
        
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

with st.expander('Sensitivity Analysis'):

    st.write('# Sensitivity Analysis (Sobol method)')

    N_samples = st.select_slider('Number of samples (N):', options=[2**i for i in range(1, 11)], value=2**3)
    display_sensitivity_df = st.checkbox("Display Sensitivity Analysis results tables", value=False)

    if st.button('Run Sensitivity Analysis'):

        st.write("✅ Defining model inputs based on the graph...")
        # Define the model inputs based on the graph G
        num_vars = G.number_of_nodes()
        names = [G.nodes[node]['label'] for node in G.nodes]
        bounds = [[0, 100] for _ in range(num_vars)]

        problem = {
            'num_vars': num_vars,
            'names': names,
            'bounds': bounds
        }
        st.write("✅ Model inputs defined")

        st.write("✅ Generating samples using the Saltelli sampler for Sobol sensitivity analysis...")
        # Generate samples using the Saltelli sampler which is more appropriate for Sobol sensitivity analysis
        param_values = saltelli.sample(problem, N_samples, calc_second_order=True)
        st.write(f"✅ Generated {len(param_values)} unique samples. Sample shape:", param_values.shape)

        st.write("✅ Initializing model results placeholder...")
        # Placeholder for model results
        Y = np.zeros([param_values.shape[0]])
        st.write("✅ Model results placeholder initialized. Placeholder shape:", Y.shape)

        st.write("✅ Defining model response wrapper function...")
        # Function to run your model
        def model_response_wrapper(param_values):
            # Convert param_values row to a dictionary {node_label: token_allocation}
            initial_tokens = {node: value for node, value in zip(G.nodes, param_values)}
            # Assuming G is globally accessible or passed as an argument
            num_steps = 50  # Define the number of steps your model runs
            log_scale = False  # Define whether to use log scale
            # Call your model function
            return model_response_sum_outcome_nodes(G, initial_tokens, num_steps, log_scale)
        st.write("✅ Model response wrapper function defined.")

        st.write("✅ Running model for each set of parameters...")
        # Run your model for each set of parameters
        progress_bar = st.progress(0)
        total_runs = len(param_values)
        for i, X in enumerate(param_values):
            Y[i] = model_response_wrapper(X)
            progress = int(((i + 1) / total_runs) * 100)
            progress_bar.progress(progress)
        st.write("✅ Model run completed for all parameter sets.")

        st.write("✅ Performing sensitivity analysis...")
        # Perform sensitivity analysis
        Si = sobol.analyze(problem, Y, print_to_console=False)
        st.write("✅ Sensitivity analysis completed.")
        # Display the first-order sensitivity indices in Streamlit

        total_Si, first_Si, second_Si = Si.to_df()

        factors_df = st.session_state.df_factors
        factors_df['OUTCOME NODE'] = factors_df['domain_name'].apply(lambda x: True if x == 'FOCAL FACTORS' else False)
        outcome_nodes = factors_df[factors_df['OUTCOME NODE']]['long_name'].tolist()

        intervenable_nodes = factors_df[factors_df['intervenable'] == 'yes']['long_name'].tolist()

        # Plotting Total Sensitivity Indices with Confidence Range as vertical bars in descending order
        total_Si_sorted = total_Si.sort_values(by='ST', ascending=True)
        fig, ax = plt.subplots(figsize=(10, 12))  # Adjusted figure size to make the plot taller
        colors = ['gold' if index in outcome_nodes else 'blue' if index in intervenable_nodes else 'grey' for index in total_Si_sorted.index]
        bars = ax.barh(total_Si_sorted.index, total_Si_sorted['ST'], xerr=total_Si_sorted['ST_conf'], color=colors, capsize=5, ecolor='grey')
        ax.set_title("Total Sensitivity Indices (ST)")
        # Set y tick labels to match bar colors
        for ticklabel, bar in zip(ax.get_yticklabels(), bars):
            ticklabel.set_color(bar.get_facecolor())
        # Adding legend
        ax.legend(handles=[matplotlib.patches.Patch(color='gold', label='Objective'),
                           matplotlib.patches.Patch(color='blue', label='Intervenable'),
                           matplotlib.patches.Patch(color='grey', label='Not Intervenable')],
                  loc='lower right')
        st.pyplot(fig)

        if display_sensitivity_df:
            st.dataframe(total_Si_sorted)

        # Plotting First Order Sensitivity Indices with Confidence Range as vertical bars in descending order
        first_Si_sorted = first_Si.sort_values(by='S1', ascending=True)
        fig, ax = plt.subplots(figsize=(10, 12))  # Adjusted figure size to make the plot taller
        colors = ['gold' if index in outcome_nodes else 'blue' if index in intervenable_nodes else 'grey' for index in first_Si_sorted.index]
        bars = ax.barh(first_Si_sorted.index, first_Si_sorted['S1'], xerr=first_Si_sorted['S1_conf'], color=colors, capsize=5, ecolor='grey')
        ax.set_title("First Order Sensitivity Indices (S1)")
        # Set y tick labels to match bar colors
        for ticklabel, bar in zip(ax.get_yticklabels(), bars):
            ticklabel.set_color(bar.get_facecolor())
        # Adding legend
        ax.legend(handles=[matplotlib.patches.Patch(color='gold', label='Objective'),
                           matplotlib.patches.Patch(color='blue', label='Intervenable'),
                           matplotlib.patches.Patch(color='grey', label='Not Intervenable')],
                  loc='lower right')
        st.pyplot(fig)

        if display_sensitivity_df:
            st.dataframe(first_Si_sorted)

        # Preparing data for heatmap
        # Extracting factor pairs and their S2 values
        factor_pairs = list(second_Si.index)
        s2_values = second_Si['S2'].values

        # Creating a matrix to fill with S2 values
        factors = set()
        for pair in factor_pairs:
            factors.update(pair)
        factors = sorted(list(factors))
        s2_matrix = pd.DataFrame(0, index=factors, columns=factors)

        # Filling the matrix with S2 values
        for pair, value in zip(factor_pairs, s2_values):
            s2_matrix.loc[pair[0], pair[1]] = value
            s2_matrix.loc[pair[1], pair[0]] = value  # Assuming symmetry

        # Plotting the heatmap
        plt.figure(figsize=(20, 20))
        mask_upper_triangle = np.triu(np.ones_like(s2_matrix, dtype=bool))
        # Mask for negative values
        mask_negative_values = s2_matrix < 0
        # Combine masks
        mask_combined = mask_upper_triangle | mask_negative_values
        ax = sns.heatmap(s2_matrix, annot=True, cmap='coolwarm', center=0, annot_kws={"size": 6}, mask=mask_combined)
        plt.title('Second Order Sensitivity Indices Heatmap (S2)', size=20)
        # Adding legend on the top right
        ax.legend(handles=[matplotlib.patches.Patch(color='gold', label='Objective'),
                           matplotlib.patches.Patch(color='blue', label='Intervenable'),
                           matplotlib.patches.Patch(color='grey', label='Not Intervenable')],
                  loc='upper right')
        # Color the x and y axis labels
        xtick_colors = ['gold' if label.get_text() in outcome_nodes else 'blue' if label.get_text() in intervenable_nodes else 'grey' for label in ax.get_xticklabels()]
        for ticklabel, color in zip(ax.get_xticklabels(), xtick_colors):
            ticklabel.set_color(color)
        ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment='right')

        ytick_colors = ['gold' if label.get_text() in outcome_nodes else 'blue' if label.get_text() in intervenable_nodes else 'grey' for label in ax.get_yticklabels()]
        for ticklabel, color in zip(ax.get_yticklabels(), ytick_colors):
            ticklabel.set_color(color)
        ax.set_yticklabels(ax.get_yticklabels(), horizontalalignment='right')
        st.pyplot(plt)

        if display_sensitivity_df:
            st.dataframe(second_Si)
