# file qlearning.py
import random
from nim import Nim

class QLearning():

    def __init__(self, alpha = 0.5, epsilon = 0.1):

        '''
            Initialize AI with an empty Q-learning dictionary,
            an alpha rate and an epsilon rate.

            The Q-learning dictionary maps '(state, action)
            pairs to a Q-value.
                - 'state' is a tuple of remaining piles, e.g. [1, 1, 4, 4]
                - 'action' is a tuple '(i, j)' for an action
        '''

        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update_model(self, old_state, action, new_state, reward):

        '''
            Update Q-learning model, given and old state, an action taken
            in that state, a new resulting state, and the reward received
            from taking that action.
        '''

        old_q = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_value(old_state, action, old_q, reward, best_future)

    def get_q_value(self, state, action):

        '''
            Return the Q-value for the state 'state' and the action 'action'.
            If no Q-value exists yet in 'self.q', return 0.
        '''

        return self.q.get((tuple(state), action), 0) 
    
    def best_future_reward(self, state):

        '''
            Given a state 'state', consider all possible '(state, action)'
            pairs available in that state and return the maximum of all
            of their Q-values.

            Use 0 as the Q-value if a '(state, action)' pair has no
            Q-value in 'self.q'. If there are no available actions
            in 'state', return 0.
        '''

        actions = list(Nim.available_actions(state))
        best_reward = float('-inf')

        for action in actions:
            current_q = self.get_q_value(state, action)
            best_reward = max(best_reward, current_q)

        return best_reward if best_reward != float('-inf') else 0

    def update_value(self, old_state, action, old_q, reward, future_rewards):

        '''
            Update the Q-value for the state 'state' and the action 'action'
            given the previous Q-value 'old_q', a current reward 'reward',
            and an estimate of future rewards 'future_rewards'.
        '''

        # Calculate the new Q-value using the Q-learning formula
        new_q = old_q + self.alpha * (reward + future_rewards - old_q)

        # Update the Q-table
        self.q[(tuple(old_state), action)] = new_q



    def choose_action(self, state, epsilon = True):

        '''
            Given a state 'state', return a action '(i, j)' to take.
            
            If 'epsilon' is 'False', then return the best action
            avaiable in the state (the one with the highest Q-value, 
            using 0 for pairs that have no Q-values).

            If 'epsilon' is 'True', then with probability 'self.epsilon'
            chose a random available action, otherwise chose the best
            action available.

            If multiple actions have the same Q-value, any of those
            options is an acceptable return value.
        '''

        best_action = None
        best_reward = float('-inf')
        actions = list(Nim.available_actions(state))
        
        # Find the best available action
        for action in actions:
            current_q = self.get_q_value(state, action)
            if best_action is None or current_q > best_reward:  # Not inclusive, since any option with the same Q-value is acceptable
                best_reward = current_q
                best_action = action
        
        # Epsilon-greedy choice    
        if epsilon and random.random() < self.epsilon :  # Verify which probability is higher to take an action
            best_action = random.choice(actions)
            
        return best_action