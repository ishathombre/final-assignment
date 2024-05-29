import torch
import torch.nn as nn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(
        self
        # Add any other arguments you need here
        # e.g. number of hidden layers, number of neurons in each layer, etc.
    ):
        """!
        Initialize a neural network. This network can be used to approximate
        some functions, maybe the reward function? Keep in mind that this class
        should only be used to define the neural network, not to train it using
        reinforcement learning algorithms.
        """

        super(NeuralNetwork, self).__init__()


    def forward(
        self, state: np.ndarray
    ) -> torch.Tensor:
        """!
        Convert the input state to the relevant output using the neural network.
        """

        return torch.zeros((1,), dtype=torch.float32)
