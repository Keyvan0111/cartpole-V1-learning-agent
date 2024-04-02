from Q_learning import *
from SARSA_learning import *



if __name__ == "__main__":
    """
    Comment out the agent you dont want to run
    """
    # Q_model = Q_agent()
    # Q_model.train_model()

    SARSA_model = SARSA_Agent()
    SARSA_model.train_model()