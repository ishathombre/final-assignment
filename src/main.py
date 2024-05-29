from environment import CatchEnv

N_EPISODES = 100

def run_environment():
    """!
    Run the environment with a random agent.
    Make sure to adapt this to train your own implementation.
    """
    env = CatchEnv()
    observation, info = env.reset()
    
    for ep in range(N_EPISODES):
        
        # Choosing a random action instead of an informed one.
        # This is where you come in!
        action = env.action_space.sample()
        # You probably want to use these values to train your agent :)
        observation, reward, terminated, truncated, info = env.step(action)

        print(f"episode {ep} | reward: {reward} | terminated: {bool(terminated)}")

        if terminated or truncated:
            observation, info = env.reset()


if __name__ == "__main__":
    run_environment()