# file sarsa.py
import random
from nim import Nim

class SARSA():

    def __init__(self, alpha = 0.5, epsilon = 0.1):

        '''
            Initialize AI with an empty SARSA dictionary,
            an alpha rate and an epsilon rate.

            The SARSA dictionary maps '(state, action)
            pairs to a Q-value.
                - 'state' is a tuple of remaining piles, e.g. [1, 1, 4, 4]
                - 'action' is a tuple '(i, j)' for an action
        '''

        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update_model(self, old_state, action, new_state, reward):

        '''
            Update SARSA model, given and old state, an action taken
            in that state, a new resulting state, and the reward received
            from taking that action.
        '''

        old_q = self.get_q_value(old_state, action)
        
        # Choose action for next state according to policy (epsilon-greedy)
        next_action = self.choose_action(new_state)
    
        # Get the Q-value of the chosen action in the new state
        future_reward = self.get_q_value(new_state, next_action)
    
        # Update the Q-value
        self.update_value(old_state, action, old_q, reward, future_reward)

    def get_q_value(self, state, action):

        '''
            Return the Q-value for the state 'state' and the action 'action'.
            If no Q-value exists yet in 'self.q', return 0.
        '''      

        return self.q.get((tuple(state), action), 0) 

    def update_value(self, old_state, action, old_q, reward, future_rewards):

        '''
            Update the Q-value for the state 'state' and the action 'action'
            given the previous Q-value 'old_q', a current reward 'reward',
            and an estimate of future rewards 'future_rewards'.
        '''

        # Calculate the new Q-value using the SARSA formula
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
    
        if not actions:
            return None
        
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