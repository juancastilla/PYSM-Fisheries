from utilities import *
from diffusion import *

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
    
    col1, col2 = st.columns(2)
    
    with col1:
        diffusion_model = st.selectbox('Choose a model:', ('one-time investment', 'continuous investment'), key='diffusion_model_1')
    with col2:
        time_horizon = st.slider('Time horizon (timesteps):', 1, 50, 25, key='time_horizon_1')


    if st.button('Run simulation', key='run_simulation_1'):

        with st.spinner('Running analysis...'):

            if diffusion_model == 'one-time investment':
                st.write("✅ Building causal diagram...")
                G = create_causal_diagram(st.session_state.df_factors, st.session_state.df_relationships)
                st.write("✅ Running simulation...")
                NEW_pulse_diffusion_network_model(G, token_dict, edited_df)
            else:
                pass

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
            time_horizon = st.slider('Time horizon (timesteps):', 1, 100, 25, key='time_horizon_compare')
        with col3: 
            rolling_window = st.slider('Rolling window (timesteps):', 1, 5, 1, key='rolling_window_compare')


        # Every form must have a submit button.
        submitted = st.form_submit_button("Run comparisons")
        if submitted:
            # if sum(token_dict_1.values()) != 100 or sum(token_dict_2.values()) != 1000:
            #     st.warning('The sum of tokens for each of the intervention packages must equal 100 to run the simulation.')
            # else:
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
            toolbox.register("mutate", custom_mutate, indpb=0.1)
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

with st.expander('Pareto Analysis'):

    st.title('Pareto Analysis')

    outcome_node_options = factors_df[factors_df['OUTCOME NODE'] == True]['long_name'].tolist()
    selected_outcome_nodes = st.multiselect('Select outcome nodes (these will be used to quantify the "Effect" objective of potential interventions):', factors_df['long_name'].tolist(), default=outcome_node_options, key='pareto_selected_outcome_nodes')
    factors_df['OUTCOME NODE'] = factors_df['long_name'].apply(lambda x: True if x in selected_outcome_nodes else False)
    st.session_state.selected_outcome_nodes = selected_outcome_nodes

    # INSERT_YOUR_CODE
    col1, col2 = st.columns(2)
    with col1:
        population_size = st.slider('NSGA2 population size (number of intervention packages):', min_value=10, max_value=200, value=50, step=10, key='population_size')
    with col2:
        num_generations = st.slider('NSGA2 generations:', min_value=0, max_value=200, value=100, step=10, key='num_generations')

    if st.button('Run optimisation', key='run_optimisation_pareto'):

        with st.spinner('Please wait...'):

            creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0, -np.inf))
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
            toolbox.register("mutate", custom_mutate, indpb=0.1)
            toolbox.register("select", tools.selNSGA2)

            # Run the genetic algorithm
            pop = toolbox.population(n=population_size)
            hof = tools.HallOfFame(10)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)

            pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=num_generations, stats=stats, halloffame=hof, verbose=True)
            st.success('Done!')
            
        # Display the Pareto solutions using Streamlit
        st.write("### Pareto Solutions")
        
        current_pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        
        # Extract the first three objectives from each individual in the Pareto front
        pareto_points = [(ind.fitness.values[0], ind.fitness.values[1], ind.fitness.values[2]) for ind in current_pareto]
        
        # Extract the first three objectives from each individual in the entire population
        pop_points = [(ind.fitness.values[0], ind.fitness.values[1], ind.fitness.values[2]) for ind in pop]
        
        # Plotting the Pareto points and population points for all combinations of objectives using Plotly
        df_pareto = pd.DataFrame(pareto_points, columns=['Control', 'Effect', 'Viability'])
        df_pop = pd.DataFrame(pop_points, columns=['Control', 'Effect', 'Viability'])
        
        objective_pairs = [(i, j) for i in range(1, 4) for j in range(i+1, 4)]
        objective_names = ['Control', 'Effect', 'Viability']
        
        # Plotting a 3D scatter plot for the three objectives using Plotly
        
        import plotly.graph_objects as go

        def format_decision_variable(node, tokens):
            return f"{st.session_state.df_factors.loc[node, 'long_name']} ({tokens})"

        text = []
        for idx, ind in enumerate(pop):
            decision_vars = [format_decision_variable(node, tokens) for node, tokens in zip(ind[:3], ind[3:])]
            ip_text = f"IP{idx+1} Decision Variables:\n<br>" + ',<br> '.join(decision_vars)
            text.append(ip_text)

        fig_3d = go.Figure()
        fig_3d.add_trace(go.Scatter3d(
            x=df_pop['Control'],
            y=df_pop['Effect'],
            z=df_pop['Viability'],
            mode='markers',
            marker=dict(size=5, color='blue', opacity=0.5),
            name='Dominated',
            hoverinfo='text',
            text = text
        ))
        fig_3d.add_trace(go.Scatter3d(
            x=df_pareto['Control'],
            y=df_pareto['Effect'],
            z=df_pareto['Viability'],
            mode='markers+text',
            marker=dict(size=8, color='red'),
            name='Pareto',
            hoverinfo='text',
            text=[f"IP{idx+1}" for idx, ind in enumerate(current_pareto)],
            textposition='middle right',
            textfont=dict(color='black')
        ))

        fig_3d.update_layout(
            title='Control vs Effect vs Viability',
            scene=dict(
                xaxis=dict(title='Control', backgroundcolor="white", gridcolor="gray", showbackground=True),
                yaxis=dict(title='Effect', backgroundcolor="white", gridcolor="gray", showbackground=True),
                zaxis=dict(title='Viability', backgroundcolor="white", gridcolor="gray", showbackground=True)
            ),
            legend_title="Legend",
            height=800
        )

        st.plotly_chart(fig_3d, use_container_width=True)
        
        import plotly.graph_objects as go
        
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
        
        for index, (i, j) in enumerate(objective_pairs):
            obj_i = objective_names[i-1]
            obj_j = objective_names[j-1]
            with columns[index]:
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_pop.iloc[:, i-1], 
                    y=df_pop.iloc[:, j-1], 
                    mode='markers', 
                    marker=dict(color='blue', size=10, opacity=0.5),
                    name='Dominated',
                    hoverinfo='text',
                    text=text
                ))
                fig.add_trace(go.Scatter(
                    x=df_pareto.iloc[:, i-1], 
                    y=df_pareto.iloc[:, j-1], 
                    mode='markers+text', 
                    marker=dict(color='red', size=12),
                    name='Pareto',
                    hoverinfo='text',
                    text=[f"IP{idx+1}" for idx, ind in enumerate(current_pareto)],
                    textposition='middle right',
                    textfont=dict(color='black')
                ))
                
                fig.update_layout(
                    title=f'{obj_i} vs {obj_j}',
                    xaxis_title=obj_i,
                    yaxis_title=obj_j,
                    legend_title="Legend"
                )
                
                st.plotly_chart(fig, use_container_width=True)



        import plotly.graph_objects as go
        import plotly.express as px  # Importing for color sequence

        # Number of Pareto individuals
        num_pareto_individuals = len(current_pareto)

        # Create a figure with subplots
        fig = go.Figure()

        # Define a color palette
        colors = px.colors.qualitative.Plotly

        # Loop through each Pareto individual
        for idx, ind in enumerate(current_pareto):
            # Nodes and their corresponding token allocations
            nodes = ind[:3]
            tokens = ind[3:]
            
            # Create a dictionary of node labels and tokens
            node_labels = [st.session_state.df_factors.loc[node, 'long_name'] for node in nodes]
            token_dict = dict(zip(node_labels, tokens))
            
            # All possible nodes in factors_df
            all_nodes = st.session_state.df_factors['long_name'].tolist()
            
            # Token values for all nodes, defaulting to 0 if not in the current individual
            token_values = [token_dict.get(node, 0) for node in all_nodes]

            # Plotting using Plotly
            fig.add_trace(go.Bar(
                y=all_nodes,
                x=token_values,
                orientation='h',
                name=f'IP {idx + 1}',
                marker=dict(color=colors[idx % len(colors)])  # Cycle through colors
            ))

        # Update layout
        fig.update_layout(
            title='Pareto Individuals Token Allocation',
            xaxis_title='Number of Tokens',
            yaxis_title='Nodes',
            barmode='stack',
            height=1000
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
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
