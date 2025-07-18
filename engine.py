import random
import matplotlib.pyplot as plt
from nim import Nim

def run_game_episode(agent, opponent=None, training=True):
    """
    Runs a single episode of the Nim game between an agent and an optional opponent.
    Each player takes turns removing objects from a pile.
    The player who removes the last object wins.

    Can be used for training or evaluation.

    Returns the winner (0 or 1).
    """
    game = Nim()
    
    # Dictionary to store the last state and action taken by each player
    last = {
        0: {"state": None, "action": None},
        1: {"state": None, "action": None}
    }

    while game.winner is None:
        state = game.piles.copy()  # Current game state

        # Determine which agent should make a move
        if opponent is not None and game.player == 1:
            action = opponent.choose_action(state)
        else:
            action = agent.choose_action(state)

        # Store the current state and action for the current player
        last[game.player] = {"state": state, "action": action}

        # Execute the move
        game.move(action)
        new_state = game.piles.copy()  # New state after move

        # If game ended, assign rewards
        if training and game.winner is not None:
            for player in (0, 1):
                # +1 if this player won, -1 if lost
                reward = 1 if game.winner == player else -1
                last_state = last[player]["state"]
                last_action = last[player]["action"]

                if last_state is not None and last_action is not None:
                    if opponent is not None and player == 1:
                        opponent.update_model(last_state, last_action, new_state, reward)
                    else:
                        agent.update_model(last_state, last_action, new_state, reward)

        # Otherwise, perform intermediate update (reward = 0)
        elif training and last[game.player]["state"] is not None:
            if opponent is not None and game.player == 1:
                opponent.update_model(last[game.player]["state"], last[game.player]["action"], new_state, 0)
            else:
                agent.update_model(last[game.player]["state"], last[game.player]["action"], new_state, 0)

    return game.winner

def train(agent, n_episodes):
    """
    Trains an agent by having it play against itself for a number of episodes.
    """
    for _ in range(n_episodes):
        run_game_episode(agent, training=True)
    return agent

def play(agent, human=None):
    """
    Allows a human player to play against the trained agent.
    """
    human = random.randint(0, 1) if human is None else human
    game = Nim()

    while game.winner is None:
        print("\nCurrent Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")

        if game.player == human:
            print("Your turn!")
            action = get_valid_user_action(game.piles)
        else:
            action = agent.choose_action(game.piles, epsilon=False)
            print(f"Agent chose to remove {action[1]} from pile {action[0]}.")

        game.move(action)

    print("\nGAME OVER")
    print("Winner:", "You" if game.winner == human else "Agent")

def get_valid_user_action(piles):
    """
    Prompts the human player to input a valid move (pile and number of objects).
    Continues prompting until a valid move is entered.
    """
    actions = Nim.available_actions(piles)
    while True:
        try:
            pile = int(input("Choose a pile: "))
            count = int(input("Number of objects to remove: "))
            if (pile, count) in actions:
                return (pile, count)
            print("Invalid move. Try again.")
        except ValueError:
            print("Please enter valid integers.")

def play_ai_vs_ai(agent1, agent2, n_games=100):
    """
    Simulates matches between two agents and plots their cumulative win counts.
    """
    wins = [0, 0]  # Win counters for agent1 and agent2
    win_progress = [[], []]  # Store win history for plotting

    for _ in range(n_games):
        winner = run_game_episode(agent1, opponent=agent2, training=False)
        if winner is not None:
            wins[winner] += 1
        for i in (0, 1):
            win_progress[i].append(wins[i])

    plot_performance(win_progress, labels=["SARSA", "Q-Learning"])

def plot_performance(win_progress, labels):
    """
    Plots the cumulative win history of two agents over multiple games.
    """
    plt.plot(win_progress[0], label=f"Agent 1 ({labels[0]})", color="blue")
    plt.plot(win_progress[1], label=f"Agent 2 ({labels[1]})", color="red")
    plt.xlabel("Game Number")
    plt.ylabel("Cumulative Wins")
    plt.title("AI vs AI Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
