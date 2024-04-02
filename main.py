from Q_learning import *
from SARSA_learning import *

if __name__ == "__main__":
    Q_model = Q_agent()
    Q_model.train_model()

    SARSA_model = SARSA_Agent()
    SARSA_model.train_model()