from abc import abstractmethod
import torch
import torch.nn as nn
from network import ConvActorCritic

device = torch.device('cpu')


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ConvActorCritic(action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor_conv.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_conv.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ConvActorCritic(action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        #self.HLoss = nn.L1Loss()

    def select_action(self, state):

        with torch.no_grad():
            if len(state) == 4:
                state = torch.FloatTensor(state)
                state = torch.transpose(state, 0, 2)
                print(state.shape)
            else:
                #print(len(state), "this is the length of state")
                if len(state) == 2:
                    state1 = torch.FloatTensor(state[0])
                else:
                    state1 = torch.FloatTensor(state)
                #print(state1.shape)
                # state = state.permute(2, 0, 1)

                state1 = state1.permute(2,0,1).unsqueeze(0)
                #print("new shape:", state1.shape)
                state = state1
            action, action_logprob, state_val = self.policy_old.act(state1)


        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action[0].item()

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        #losses = []
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards to have a mean of 0 and a standard deviation of 1.
        # This reduces the variance in reward signals, allowing for more consistent and efficient policy updates.
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # Calculate advantage estimates
        
        # it estimates how good an action is, compared to average action for a specific state
        # If the advantage function is positive, it means that the action taken by the agent is good
        # and that we can have good reward by taking the action. The idea here is to improve those actions probability
        # On the other hand, if the advantage is negative, then we need to decrease the action probabilities

        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values

            #print("i am here now!")
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss --> compute clipped loss gradients
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            #losses.append(loss.mean().item())
            #print("loss:", losses)

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        return loss.mean().item()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
