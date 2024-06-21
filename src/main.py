import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import os
from agent import PPO
from datetime import datetime
from environment import CatchEnv
import pandas as pd

env_name = "Catch"
env = CatchEnv()

# state space dimension
state_dim = env.observation_space.shape[0]

# action space dimension
action_dim = env.action_space.n

####### initialize environment hyperparameters ######

max_ep_len = 440  # max timesteps in one episode --> these are 40 episodes, we do that since we are not expliciting the batch size
max_training_timesteps = 110000  # break training loop if timeteps > max_training_timesteps (330000 = 30K EPISODES)

log_freq = max_ep_len * 2  # log avg reward in the interval (every 80 episodes in num timesteps)
save_model_freq = 22000  # save model frequency (every 22000 timesteps or 2K episodes)

#update_timestep = max_ep_len * 20  # update policy every 800 episodes
update_timestep = max_ep_len * 2  # update policy every 80 episodes
#update_timestep = max_ep_len  # update policy every 40 episodes

K_epochs = 40  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
#eps_clip = 0.4
#eps_clip = 0.7
gamma = 0.99  # discount factor

lr_actor = 0.0003  
lr_critic = 0.0007

## BEST TIL NOW, 0.93 REWARD IN 20K EPISODES

#lr_actor = 0.0003  # learning rate for actor network
#lr_critic = 0.0007  # learning rate for critic network, it should be larger than actor lr

#lr_actor = 0.0005  # same lr
#lr_critic = 0.0005

#lr_actor = 0.0001
#lr_critic = 0.0003


random_seed = 0  # set random seed if required (0 = no random seed)

############# print all hyperparameters #############

print("--------------------------------------------------------------------------------------------")

print("max training timesteps : ", max_training_timesteps)
print("max timesteps per episode : ", max_ep_len)

print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")

print("--------------------------------------------------------------------------------------------")

print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)

print("--------------------------------------------------------------------------------------------")

print("Initializing a discrete action space policy")

print("--------------------------------------------------------------------------------------------")

print("PPO update frequency : " + str(update_timestep) + " timesteps")
print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)

print("--------------------------------------------------------------------------------------------")

print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)

if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)


###################### logging ######################

#### log files for multiple runs are NOT overwritten

log_dir = "PPO_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#### get number of log files in log directory
run_num = 0
current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)

#### create new log file for each run
log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

print("current logging run number for " + env_name + " : ", run_num)
print("logging at : " + log_f_name)

#####################################################

################### checkpointing ###################

run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

directory = "PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)

#####################################################


def run_environment():
    print("============================================================================================")

    ################################### Training ###################################

    print("training environment name : " + env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    action_dim = env.action_space.n

    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    loss_ = 0
    losses_ = []

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        # print("this is state", state)
        # print(state.shape)

        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):

            # select action with policy

            # print("this is also state", state)
           
            action = ppo_agent.select_action(state)
            print(action, "this is the action!")
            state, reward, done, _, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                loss_ = ppo_agent.update()
                losses_.append(loss_)

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0


            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is finished
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
        df = pd.DataFrame(losses_)
        df.to_csv('losses_.csv', index=False)

        print(f"episode {i_episode} | reward: {reward} | terminated: {bool(done)}")

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    return losses_


if __name__ == "__main__":
   run_environment()
