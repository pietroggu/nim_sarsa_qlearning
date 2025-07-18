from sarsa import SARSA
from qlearning import QLearning
from engine import train, play, play_ai_vs_ai

def main():
    # Initialize and train agents
    sarsa_agent = train(SARSA(), n_episodes=10000)
    q_learning_agent = train(QLearning(), n_episodes=10000)

    # Evaluate agents against each other
    print("\n--- AI vs AI: SARSA vs Q-Learning ---")
    play_ai_vs_ai(sarsa_agent, q_learning_agent, n_games=100)

    # Let the human play against each agent
    print("\n--- Play against the SARSA Agent ---")
    play(sarsa_agent)

    print("\n--- Play against the Q-Learning Agent ---")
    play(q_learning_agent)

if __name__ == "__main__":
    main()
