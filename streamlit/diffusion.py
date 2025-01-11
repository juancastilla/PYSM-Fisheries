import logging
import warnings
import matplotlib

# Suppress specific matplotlib warnings
logging.getLogger('matplotlib.font_manager').disabled = True
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# Also set a default font that we know exists
matplotlib.rcParams['font.sans-serif'] = ['Liberation Sans', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['font.family'] = 'sans-serif'

import networkx as nx
import matplotlib.pyplot as plt
import random
from enum import Enum
from collections import defaultdict
from pyvis.network import Network
from IPython.display import IFrame
import os
import logging
import json
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CausalTokenModel')

# Token polarity and delays

class Polarity(Enum):
    SAME = 1
    OPPOSITE = -1

class Strength(Enum):
    WEAK = 0.3
    MEDIUM = 0.6
    HIGH = 1.0

class Delay(Enum):
    FAST = 1
    MEDIUM = 3
    SLOW = 10

class TokenState(Enum):
    IN_TRANSIT = "in_transit"
    READY = "ready"
    ACCUMULATED = "accumulated"

class NodeType(Enum):
    PASS_THROUGH = "pass_through"
    CONSUME = "consume"
    ACCUMULATE = "accumulate"

class TokenAgent:
    """An agent representing a token that diffuses through the causal network."""
    
    def __init__(self, unique_id, model, initial_node, initial_value=1.0, initial_charge=1):
        self.unique_id = unique_id
        self.model = model
        self.current_node = initial_node
        self.value = initial_value
        self.charge = initial_charge  # Initialize with specified charge
        self.state = TokenState.READY
        self.transit_steps_remaining = 0
        self.target_node = None
        self.active = True
        logger.info(f"Token {unique_id} initialized at node {initial_node} with charge {initial_charge}")
    
    def start_movement(self, target_node, edge_data):
        """Begin movement to a new node."""
        self.state = TokenState.IN_TRANSIT
        self.target_node = target_node
        self.transit_steps_remaining = edge_data['delay'].value
        # Update charge based on edge polarity
        if edge_data['polarity'] == Polarity.OPPOSITE:
            self.charge *= -1  # Flip charge if polarity is OPPOSITE
        logger.debug(f"Token {self.unique_id} starting movement from {self.current_node} to {target_node} with delay {edge_data['delay'].value} steps and charge {self.charge}")
    
    def complete_movement(self):
        """Complete movement to target node."""
        logger.debug(f"Token {self.unique_id} completed movement to {self.target_node}")
        self.current_node = self.target_node
        self.target_node = None
        self.transit_steps_remaining = 0
    
    def get_outgoing_strengths(self):
        """Get all outgoing edges and their normalized strengths."""
        outgoing_edges = []
        total_strength = 0
        
        # Collect edges and calculate total strength
        for neighbor in self.model.G.neighbors(self.current_node):
            edge_data = self.model.G.edges[self.current_node, neighbor]
            strength = edge_data['strength'].value
            outgoing_edges.append((neighbor, edge_data))
            total_strength += strength
        
        if not outgoing_edges or total_strength == 0:
            logger.debug(f"Token {self.unique_id} has no valid outgoing edges from {self.current_node}")
            return []
        
        # Calculate normalized strengths
        edge_strengths = []
        for neighbor, edge_data in outgoing_edges:
            normalized_strength = edge_data['strength'].value / total_strength
            edge_strengths.append((neighbor, edge_data, normalized_strength))
        
        return edge_strengths
    
    def step(self):
        """Step function - tokens remain active at terminal nodes."""
        if self.state == TokenState.IN_TRANSIT:
            self.transit_steps_remaining -= 1
            logger.debug(f"Token {self.unique_id} has {self.transit_steps_remaining} steps remaining on edge {self.current_node}->{self.target_node}")
            if self.transit_steps_remaining <= 0:
                self.complete_movement()
                self.state = TokenState.READY
        
        elif self.state == TokenState.READY:
            # Check if current node is accumulating and not at capacity
            node_data = self.model.G.nodes[self.current_node]
            if (node_data.get('type') == NodeType.ACCUMULATE and 
                node_data.get('accumulated_tokens', 0) < node_data.get('accumulation_capacity', 0)):
                self.state = TokenState.ACCUMULATED
                node_data['accumulated_tokens'] = node_data.get('accumulated_tokens', 0) + 1
                logger.debug(f"Token {self.unique_id} accumulated at node {self.current_node}")
                return

            edge_strengths = self.get_outgoing_strengths()
            if edge_strengths:  # Only move if there are outgoing edges
                # Choose destination based on strengths
                target_node, edge_data, _ = random.choices(
                    edge_strengths, 
                    weights=[strength for _, _, strength in edge_strengths],
                    k=1
                )[0]
                self.start_movement(target_node, edge_data)

class CausalTokenModel:
    """A simplified model for token diffusion in a causal loop diagram."""
    
    def __init__(self, G, num_tokens=10, initial_allocation=None):
        self.G = G
        self.num_tokens = num_tokens
        self.agents = []
        self.edge_flows_over_time = []
        self.node_flows_over_time = []
        self.step_count = 0
        self.simulation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize accumulated_tokens counter for accumulation nodes
        for node, data in self.G.nodes(data=True):
            if data.get('type') == NodeType.ACCUMULATE:
                data['accumulated_tokens'] = 0
        
        logger.info(f"Initializing simulation {self.simulation_id} with {num_tokens} tokens")
        
        # Initialize tokens according to initial_allocation or default to node 'A'
        if initial_allocation is None:
            initial_allocation = {'0': num_tokens}
            
        token_id = 0
        for node, count in initial_allocation.items():
            for _ in range(count):
                token = TokenAgent(token_id, self, node)
                self.agents.append(token)
                token_id += 1
        
        # Record initial node flows
        self.node_flows_over_time.append(self.get_node_flows())
        
        # Log initial graph configuration
        self._log_graph_config()
    
    def _log_graph_config(self):
        """Log the initial graph configuration."""
        config = {
            "nodes": [{
                "id": node,
                "type": str(self.G.nodes[node].get('type', NodeType.PASS_THROUGH)),
                "consumption_rate": self.G.nodes[node].get('consumption_rate', 0),
                "accumulation_capacity": self.G.nodes[node].get('accumulation_capacity', 0)
            } for node in self.G.nodes()],
            "edges": [{
                "source": u,
                "target": v,
                "polarity": str(data['polarity']),
                "strength": str(data['strength']),
                "delay": str(data['delay'])
            } for u, v, data in self.G.edges(data=True)]
        }
        logger.info(f"Graph configuration: {json.dumps(config, indent=2)}")
    
    def get_edge_flows(self):
        """Calculate current flow distribution on edges."""
        flows = {}
        for edge in self.G.edges(data=True):
            source, target = edge[0], edge[1]
            pos_count = sum(1 for agent in self.agents 
                          if agent.state == TokenState.IN_TRANSIT 
                          and agent.current_node == source 
                          and agent.target_node == target
                          and agent.charge > 0)
            neg_count = sum(1 for agent in self.agents 
                          if agent.state == TokenState.IN_TRANSIT 
                          and agent.current_node == source 
                          and agent.target_node == target
                          and agent.charge < 0)
            flows[f"{source}->{target}"] = (pos_count, neg_count)
        return flows
    
    def get_node_flows(self):
        """Calculate current token distribution on nodes with charge information."""
        flows = defaultdict(lambda: {'positive': 0, 'negative': 0, 'accumulated': 0})
        for agent in self.agents:
            if agent.state in [TokenState.READY, TokenState.ACCUMULATED]:
                if agent.state == TokenState.ACCUMULATED:
                    flows[agent.current_node]['accumulated'] += 1
                elif agent.charge > 0:
                    flows[agent.current_node]['positive'] += 1
                else:
                    flows[agent.current_node]['negative'] += 1
        return dict(flows)
    
    def step(self):
        """Advance the model by one step, consuming tokens at nodes."""
        self.step_count += 1
        logger.info(f"Starting step {self.step_count}")
        
        # Log current state before step
        logger.debug(f"Current node flows: {json.dumps(self.get_node_flows())}")
        logger.debug(f"Current edge flows: {json.dumps(self.get_edge_flows())}")
        
        # Process nodes based on their type
        for node in self.G.nodes(data=True):
            node_id = node[0]
            node_type = node[1].get('type', NodeType.PASS_THROUGH)
            
            if node_type == NodeType.CONSUME:
                consumption_rate = node[1].get('consumption_rate', 0)
                agents_at_node = [agent for agent in self.agents if agent.current_node == node_id and agent.state == TokenState.READY]
                
                # Consume tokens at specified rate
                consumed_count = 0
                for _ in range(int(consumption_rate)):
                    if agents_at_node:
                        agent_to_consume = agents_at_node.pop(0)
                        self.agents.remove(agent_to_consume)
                        consumed_count += 1
                
                if consumed_count > 0:
                    logger.info(f"Consumed {consumed_count} tokens at node {node_id}")
        
        # Move remaining agents
        for agent in self.agents:
            agent.step()
        
        self.edge_flows_over_time.append(self.get_edge_flows())
        self.node_flows_over_time.append(self.get_node_flows())
        
        # Log summary of step
        logger.info(f"Step {self.step_count} complete. {len(self.agents)} tokens remaining")
    
    def plot_edge_flows(self, df_factors, df_relationships):
        """Plot the edge flows over time with charge information."""
        if not self.edge_flows_over_time:
            logger.warning("No edge flow data available for plotting")
            return
            
        edges = list(self.edge_flows_over_time[0].keys())
        num_edges = len(edges)
        
        # Calculate number of rows needed (half of num_edges, rounded up)
        num_rows = (num_edges + 1) // 2
        
        # Reduce figure size to avoid decompression bomb error
        fig = plt.figure(figsize=(12, 3*num_rows))
        
        for i, edge in enumerate(edges, 1):
            # Parse the edge name to get the source and target nodes
            source, target = map(int, edge.split('->'))
            # Get the source and target names from df_factors, short_name column
            source_name = df_factors[df_factors['factor_id'] == source]['short_name'].values[0]
            target_name = df_factors[df_factors['factor_id'] == target]['short_name'].values[0]
            # Get the polarity from df_relationships
            polarity = df_relationships[(df_relationships['from_factor_id'] == source) & 
                                      (df_relationships['to_factor_id'] == target)]['polarity'].values[0]
            # Add polarity to title
            polarity_symbol = 'same' if polarity == 'positive' else 'opposite'

            # Calculate subplot position in 2 columns
            ax = fig.add_subplot(num_rows, 2, i)
            
            pos_values = [flow[edge][0] for flow in self.edge_flows_over_time]
            neg_values = [-flow[edge][1] for flow in self.edge_flows_over_time] # Negate for plotting below x-axis
            
            # Plot bars with reduced dpi
            x = range(len(pos_values))
            ax.bar(x, pos_values, color='g', label='Target Increase')
            ax.bar(x, neg_values, color='r', label='Target Decrease')
            
            # Use smaller font sizes
            ax.set_title(f'{source_name} ---[{polarity_symbol}]---> {target_name}', fontsize=8)
            ax.set_xlabel('Time Step', fontsize=8)
            ax.set_ylabel('Number of Tokens in Transit', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=8)
            ax.grid(True)
            
        plt.tight_layout()
        # Save with reduced DPI and quality
        st.pyplot(fig, dpi=150)
        logger.info("Edge flow plots displayed in Streamlit")


    def plot_node_flows(self):
        """Plot the node flows over time with charge information."""
        if not self.node_flows_over_time:
            logger.warning("No node flow data available for plotting")
            return
            
        nodes = list(self.G.nodes())
        num_nodes = len(nodes)
        
        # Calculate number of rows needed (half of num_nodes, rounded up)
        num_rows = (num_nodes + 1) // 2
        
        fig = plt.figure(figsize=(15, 4*num_rows))
        
        for i, node in enumerate(nodes, 1):
            pos_values = [flow.get(node, {'positive':0, 'negative':0, 'accumulated':0})['positive'] 
                         for flow in self.node_flows_over_time]
            neg_values = [-flow.get(node, {'positive':0, 'negative':0, 'accumulated':0})['negative'] 
                         for flow in self.node_flows_over_time]  # Negate for plotting below x-axis
            acc_values = [flow.get(node, {'positive':0, 'negative':0, 'accumulated':0})['accumulated'] 
                         for flow in self.node_flows_over_time]
            
            # Calculate subplot position in 2 columns
            row = (i-1) // 2
            col = (i-1) % 2
            ax = fig.add_subplot(num_rows, 2, i)
            
            # Plot bars
            x = range(len(pos_values))
            ax.bar(x, pos_values, color='g', label='Increase')
            ax.bar(x, neg_values, color='r', label='Decrease')
            
            # Plot accumulated line
            ax.plot(x, acc_values, 'b--', label='Accumulation', linewidth=2)
            
            # Use node label instead of node number
            node_label = self.G.nodes[node]['label']
            ax.set_title(f'{node_label}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Number of Tokens')
            ax.legend()
            ax.grid(True)
            
        plt.tight_layout()
        st.pyplot(fig)
        logger.info("Node flow plots displayed in Streamlit")

    def plot_net_effects(self):
        """Plot net effects (positive - negative) and cumulative effects over time."""
        if not self.node_flows_over_time:
            logger.warning("No node flow data available for plotting")
            return

        nodes = list(self.G.nodes())
        num_nodes = len(nodes)
        
        # Calculate number of rows needed (half of num_nodes, rounded up)
        num_rows = (num_nodes + 1) // 2

        fig = plt.figure(figsize=(15, 4*num_rows))

        for i, node in enumerate(nodes, 1):
            # Calculate subplot position in 2 columns
            row = (i-1) // 2
            col = (i-1) % 2
            ax = fig.add_subplot(num_rows, 2, i)
            
            # Calculate net effect (positive - negative)
            net_values = [flow.get(node, {'positive':0, 'negative':0})['positive'] - 
                         flow.get(node, {'positive':0, 'negative':0})['negative']
                         for flow in self.node_flows_over_time]
            
            # Calculate cumulative effect
            cumulative_values = []
            running_sum = 0
            for net_val in net_values:
                running_sum += net_val
                cumulative_values.append(running_sum)

            x = range(len(net_values))
            
            # Plot net effect bars with colors based on value
            colors = ['g' if val >= 0 else 'r' for val in net_values]
            # Create two bar plots for legend
            ax.bar(x, [val if val >= 0 else 0 for val in net_values], color='g', label='Net Effect — Increase')
            ax.bar(x, [val if val < 0 else 0 for val in net_values], color='r', label='Net Effect — Decrease')
            
            # Plot cumulative effect line
            ax.plot(x, cumulative_values, 'b-', label='Cumulative Effect', linewidth=2)
            
            # Use node label instead of node number
            node_label = self.G.nodes[node]['label']
            ax.set_title(f'{node_label}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Token Effect')
            ax.legend()
            ax.grid(True)
            
            # Add zero line for reference
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig)
        logger.info("Net effects plots displayed in Streamlit")

    def plot_outcome_net_effects(self, edited_df):
        """Plot net effects (positive - negative) and cumulative effects over time for outcome nodes only."""
        if not self.node_flows_over_time:
            logger.warning("No node flow data available for plotting")
            return

        # Get outcome nodes from df_factors
        outcome_node_ids = edited_df[edited_df['OUTCOME NODE'] == True]['factor_id'].tolist()
        
        if not outcome_node_ids:
            logger.warning("No outcome nodes found in the dataframe")
            return

        num_nodes = len(outcome_node_ids)
        # Calculate number of rows needed (half of num_nodes, rounded up)
        num_rows = (num_nodes + 1) // 2

        fig = plt.figure(figsize=(15, 4*num_rows))

        for i, node in enumerate(outcome_node_ids, 1):
            # Calculate subplot position in 2 columns
            row = (i-1) // 2
            col = (i-1) % 2
            ax = fig.add_subplot(num_rows, 2, i)
            
            # Calculate net effect (positive - negative)
            net_values = [flow.get(node, {'positive':0, 'negative':0})['positive'] - 
                         flow.get(node, {'positive':0, 'negative':0})['negative']
                         for flow in self.node_flows_over_time]
            
            # Calculate cumulative effect
            cumulative_values = []
            running_sum = 0
            for net_val in net_values:
                running_sum += net_val
                cumulative_values.append(running_sum)

            x = range(len(net_values))
            
            # Plot net effect bars with colors based on value
            ax.bar(x, [val if val >= 0 else 0 for val in net_values], color='g', label='Net Effect — Increase')
            ax.bar(x, [val if val < 0 else 0 for val in net_values], color='r', label='Net Effect — Decrease')
            
            # Plot cumulative effect line
            ax.plot(x, cumulative_values, 'b-', label='Cumulative Effect', linewidth=2)
            
            # Use node label instead of node number
            node_label = self.G.nodes[node]['label']
            ax.set_title(f'{node_label}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Token Effect')
            ax.legend()
            ax.grid(True)
            
            # Add zero line for reference
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig)
        logger.info("Net effects plots for outcome nodes displayed in Streamlit")

def run_simulation(G, num_tokens=10, num_steps=30, initial_allocation=None):
    """Run the token diffusion simulation and return results."""
    logger.info(f"Starting simulation with {num_tokens} tokens for {num_steps} steps")
    model = CausalTokenModel(G, num_tokens, initial_allocation)
    
    for step in range(num_steps):
        if step == 0:  # Only count first decisions
            for agent in model.agents:
                if agent.state == TokenState.READY:
                    edge_strengths = agent.get_outgoing_strengths()
                    if edge_strengths:
                        target_node, _, _ = random.choices(
                            edge_strengths,
                            weights=[strength for _, _, strength in edge_strengths],
                            k=1
                        )[0]
            
        model.step()
    
    logger.info("Simulation complete")
    return model

def create_causal_diagram(df_factors, df_relationships):
    G = nx.DiGraph()
    
    # Add nodes with type and consumption rate attributes
    nodes = []
    for _, row in df_factors.iterrows():
        node_attrs = {
            'type': getattr(NodeType, row['node_type']),
            'label': row['short_name']  # Add short_name as label attribute
        }
        
        if row['node_type'] == 'PASS_THROUGH':
            node_attrs.update({'consumption_rate': 0})
        elif row['node_type'] == 'CONSUME':
            node_attrs.update({'consumption_rate': row['consume_rate_tokens']})
        elif row['node_type'] == 'ACCUMULATE':
            node_attrs.update({
                'consumption_rate': 0,
                'accumulation_capacity': int(row['accum_rate_tokens'])
            })
            
        nodes.append((row['factor_id'], node_attrs))
    
    G.add_nodes_from(nodes)
    
    # Add edges with properties
    edges = []
    for _, row in df_relationships.iterrows():
        edge_attrs = {
            'polarity': Polarity.SAME if row['polarity'] == 'positive' else Polarity.OPPOSITE,
            'strength': Strength.HIGH if row['strength'] == 'strong' else Strength.MEDIUM if row['strength'] == 'medium' else Strength.LOW,
            'delay': Delay.FAST if row['delay'] == 'fast' else Delay.MEDIUM if row['delay'] == 'medium' else Delay.SLOW
        }
        edges.append((row['from_factor_id'], row['to_factor_id'], edge_attrs))
    G.add_edges_from(edges)
    
    return G

def NEW_pulse_diffusion_network_model(G, initial_tokens, edited_df):

    model = run_simulation(G, num_tokens=100, num_steps=100, initial_allocation=initial_tokens)


    # Calculate the percentage of nodes with non-zero token counts from model.node_flows_over_time
    # Convert list of node flows to a format we can analyze
    node_flows_df = pd.DataFrame(model.node_flows_over_time)
    
    st.write("✅ Calculating system controllability...")
    # Calculate percentage of nodes that had tokens at any point
    nodes_with_tokens = sum(1 for node in G.nodes() if any(
        flow.get(node, {'positive': 0, 'negative': 0, 'accumulated': 0})['positive'] > 0 or
        flow.get(node, {'positive': 0, 'negative': 0, 'accumulated': 0})['negative'] > 0 or 
        flow.get(node, {'positive': 0, 'negative': 0, 'accumulated': 0})['accumulated'] > 0
        for flow in model.node_flows_over_time
    ))
    non_zero_tokens_percentage = (nodes_with_tokens / len(G.nodes())) * 100

    st.markdown("<h2 style='text-align: center;'>% System Controlability</h2>", unsafe_allow_html=True)

    # Create a gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=non_zero_tokens_percentage,
        number={'suffix': "%"},  # Add a '%' suffix to the number displayed
        gauge={'axis': {'range': [None, 100]}}
    ))

    # Display the gauge chart in Streamlit
    st.plotly_chart(fig, use_container_width=True, key="gauge_chart1")

    st.write("✅ Plotting outcome net effects...")
    st.markdown("<h2 style='text-align: center;'>Net Effects Over Time on Outcome Nodes</h2>", unsafe_allow_html=True)
    model.plot_outcome_net_effects(edited_df)

    st.write("✅ Plotting edge flows...")
    st.markdown("<h2 style='text-align: center;'>Edge Flows Over Time</h2>", unsafe_allow_html=True)
    model.plot_edge_flows(st.session_state.df_factors, st.session_state.df_relationships)
    
    st.write("✅ Plotting node flows...")
    st.markdown("<h2 style='text-align: center;'>Node Flows Over Time</h2>", unsafe_allow_html=True)
    model.plot_node_flows()
    
    st.write("✅ Plotting net effects...")
    st.markdown("<h2 style='text-align: center;'>Net Effects Over Time</h2>", unsafe_allow_html=True)
    model.plot_net_effects()