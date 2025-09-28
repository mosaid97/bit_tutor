# services/recommendation/models/rl_recommender_agent.py

import random
import numpy as np

class RL_Recommender_Agent:
    """
    A Reinforcement Learning (RL) agent that learns the optimal policy for
    recommending learning activities to a student.
    """
    def __init__(self, kcs_map, exercises, action_space_size):
        self.kcs_map = kcs_map
        self.exercises = exercises
        self.action_space_size = action_space_size
        # Q-table: state -> array of action values. A simple form of policy.
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.max_exploration_rate = 1.0
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.01
        print("Initialized Reinforcement Learning (RL) Agent.")

    def get_state(self, student_graph_obj):
        """Converts the student's mastery profile into a discrete state for the Q-table."""
        mastery_levels = [
            student_graph_obj.graph.edges[student_graph_obj.student_id, kc]['mastery']
            for kc in sorted(self.kcs_map.keys())
        ]
        # Discretize the continuous mastery levels into a tuple state
        state = tuple(int(m * 10) for m in mastery_levels) # e.g., (1, 2, 1, 3,...)
        return state

    def choose_action(self, state):
        """
        Chooses an action using an epsilon-greedy strategy: either explore
        a random action or exploit the best-known action from the Q-table.
        """
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.randint(0, self.action_space_size - 1) # Explore
        else:
            # Exploit: choose the best action from the current policy
            action_values = self.q_table.get(state, np.zeros(self.action_space_size))
            action = np.argmax(action_values)
        return action

    def update_policy(self, state, action, reward, new_state):
        """
        Updates the Q-table using the Bellman equation. This is where the agent "learns".
        """
        old_value = self.q_table.get(state, np.zeros(self.action_space_size))[action]
        
        future_optimal_value = np.max(self.q_table.get(new_state, np.zeros(self.action_space_size)))
        
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * future_optimal_value - old_value)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size)
        self.q_table[state][action] = new_value

    def decay_exploration_rate(self, episode):
        """Reduces the exploration rate over time to shift from exploration to exploitation."""
        self.exploration_rate = self.min_exploration_rate + \
            (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)

    def get_best_action(self, state):
        """
        Get the best action for a given state without exploration.
        Used for inference after training.
        """
        if state not in self.q_table:
            return random.randint(0, self.action_space_size - 1)
        
        action_values = self.q_table[state]
        return np.argmax(action_values)

    def get_action_values(self, state):
        """
        Get the Q-values for all actions in a given state.
        Useful for analysis and debugging.
        """
        return self.q_table.get(state, np.zeros(self.action_space_size))

    def save_policy(self, filepath):
        """
        Save the learned Q-table to a file.
        
        Args:
            filepath (str): Path to save the policy
        """
        try:
            import pickle
            with open(filepath, 'wb') as f:
                policy_data = {
                    'q_table': self.q_table,
                    'kcs_map': self.kcs_map,
                    'exercises': self.exercises,
                    'action_space_size': self.action_space_size,
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'exploration_rate': self.exploration_rate
                }
                pickle.dump(policy_data, f)
            print(f"RL policy saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving RL policy: {e}")
            return False

    @classmethod
    def load_policy(cls, filepath):
        """
        Load a saved Q-table from a file.
        
        Args:
            filepath (str): Path to the saved policy
            
        Returns:
            RL_Recommender_Agent: Loaded agent with trained policy
        """
        try:
            import pickle
            with open(filepath, 'rb') as f:
                policy_data = pickle.load(f)
            
            # Create new agent instance
            agent = cls(
                policy_data['kcs_map'],
                policy_data['exercises'],
                policy_data['action_space_size']
            )
            
            # Restore the learned policy
            agent.q_table = policy_data['q_table']
            agent.learning_rate = policy_data.get('learning_rate', 0.1)
            agent.discount_factor = policy_data.get('discount_factor', 0.9)
            agent.exploration_rate = policy_data.get('exploration_rate', 0.01)
            
            print(f"RL policy loaded from {filepath}")
            print(f"Loaded Q-table with {len(agent.q_table)} states")
            return agent
        except Exception as e:
            print(f"Error loading RL policy: {e}")
            return None

    def get_policy_stats(self):
        """
        Get statistics about the learned policy.
        
        Returns:
            dict: Statistics about the Q-table and policy
        """
        if not self.q_table:
            return {
                'num_states': 0,
                'total_state_action_pairs': 0,
                'avg_q_value': 0.0,
                'max_q_value': 0.0,
                'min_q_value': 0.0
            }
        
        all_q_values = []
        for state_actions in self.q_table.values():
            all_q_values.extend(state_actions)
        
        return {
            'num_states': len(self.q_table),
            'total_state_action_pairs': len(all_q_values),
            'avg_q_value': np.mean(all_q_values) if all_q_values else 0.0,
            'max_q_value': np.max(all_q_values) if all_q_values else 0.0,
            'min_q_value': np.min(all_q_values) if all_q_values else 0.0,
            'exploration_rate': self.exploration_rate
        }
