import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ConvActorCritic(nn.Module):
    def __init__(self, action_dim):
        super(ConvActorCritic, self).__init__()

        # actor
        self.actor_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.actor_fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.critic_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.critic_fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # actor forward pass
        actor_conv_out = self.actor_conv(x)
        actor_conv_out = actor_conv_out.view(actor_conv_out.size(0), -1)
        actor_out = self.actor_fc(actor_conv_out)

        # critic forward pass
        critic_conv_out = self.critic_conv(x)
        critic_conv_out = critic_conv_out.view(critic_conv_out.size(0), -1)
        critic_out = self.critic_fc(critic_conv_out)

        return actor_out, critic_out

    def act(self, state):
        actor_conv_out = self.actor_conv(state)
        actor_conv_out = actor_conv_out.view(actor_conv_out.size(0), -1)
        actor_probs = self.actor_fc(actor_conv_out)
        dist = Categorical(actor_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        critic_conv_out = self.critic_conv(state)
        critic_conv_out = critic_conv_out.view(critic_conv_out.size(0), -1)
        state_val = self.critic_fc(critic_conv_out)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        actor_conv_out = self.actor_conv(state)
        actor_conv_out = actor_conv_out.view(actor_conv_out.size(0), -1)
        action_probs = self.actor_fc(actor_conv_out)
        dist = Categorical(action_probs)


        dist_entropy = dist.entropy()
        action_logprobs = dist.log_prob(action)
        critic_conv_out = self.critic_conv(state)
        critic_conv_out = critic_conv_out.view(critic_conv_out.size(0), -1)
        state_values = self.critic_fc(critic_conv_out)

        return action_logprobs, state_values, dist_entropy



# Example
state_dim = (4, 84, 84)
action_dim = 3
model = ConvActorCritic(action_dim)

# Dummy input tensor with shape (batch_size, 4, 84, 84)
input_tensor = torch.randn(1, 4, 84, 84)
print(input_tensor.shape)
actor_output, critic_output = model(input_tensor)

print(actor_output)
print(critic_output)
