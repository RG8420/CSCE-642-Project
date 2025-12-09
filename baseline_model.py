import numpy as np
import pandas as pd
from flexsim_env_v2025 import FlexSimEnv 
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env

# *** ADD THIS IMPORT ***
# from stable_baselines3.common.policies import LstmPolicy

from gymnasium import ActionWrapper
from gymnasium.spaces import Box

# Define the bounds of your MultiDiscrete action space
# N_i = number of options for each discrete choice
DISCRETE_BOUNDS = np.array([15, 15, 15, 3, 2, 2, 2])

class ContinuousToDiscreteWrapper(ActionWrapper):
    """
    Wraps a MultiDiscrete environment to accept continuous actions (Box space).
    """
    def __init__(self, env):
        super().__init__(env)
        # 1. Define the new continuous action space (Box)
        # It must have the same shape as the MultiDiscrete array (7 dimensions)
        self.action_space = Box(low=-1.0, high=1.0, shape=(len(DISCRETE_BOUNDS),), dtype=np.float32)
        
        # 2. Store the original discrete bounds for mapping
        self.discrete_bounds = DISCRETE_BOUNDS

    def action(self, continuous_action: np.ndarray) -> np.ndarray:
        """
        Converts a continuous action array (e.g., [-0.5, 0.9, 0.1, ...]) 
        to a discrete action array (e.g., [7, 14, 8, ...]).
        
        :param continuous_action: An array of floats between -1.0 and 1.0.
        :return: An array of integers representing the discrete action indices.
        """
        converted_discrete_action = np.empty_like(continuous_action, dtype=np.int32)
        
        # Calculate N - 1, which is the maximum index for each dimension
        max_indices = self.discrete_bounds - 1

        for i in range(len(continuous_action)):
            
            # --- Step 1: Scale and Shift ---
            # Map [-1.0, 1.0] to [0.0, 1.0]: (x + 1) / 2
            normalized_val = (continuous_action[i] + 1.0) / 2.0
            
            # Map [0.0, 1.0] to [0.0, max_index]: normalized_val * max_indices[i]
            scaled_index = normalized_val * max_indices[i]
            
            # --- Step 2: Round and Step 3: Clip ---
            # Round to the nearest integer and clip to ensure it stays in the valid range [0, N-1]
            discrete_index = np.clip(
                np.round(scaled_index),
                0,
                max_indices[i]
            ).astype(np.int32)
            
            converted_discrete_action[i] = discrete_index

        # The result must be a NumPy array of integers
        return converted_discrete_action

def main():
    print("Initializing FlexSim environment...")

    # Create a FlexSim OpenAI Gym Environment
    env = FlexSimEnv(
        flexsimPath = r"D:\Softwares\flexsim2024\program\flexsim.exe",
        modelPath = r"D:\Projects\CSCE642_DRL\Project\FJSSP_v2_15_Types_3_Machines_4_Operations_v7.fsm",
        verbose = False,
        visible = False
        )
    check_env(env) # Check that an environment follows Gym API.1
    env = ContinuousToDiscreteWrapper(env)

    # Training a baselines3 PPO model in the environment
    # model = PPO("MlpPolicy", env, verbose=0)
    # model = PPO(LstmPolicy, env=env, verbose=0)
    model = SAC("MlpPolicy", env, verbose=0)
    print("Training model...")
    model.learn(total_timesteps=1000)
    
    # save the model
    print("Saving model...")
    model.save("MyTrainedModel_SAC_ThesisRLModel_Breakdown")

    input("Waiting for input to do some test runs...")

    # Run test episodes using the trained model
    EPOCHS = 100
    cumulative_rewards = []
    for i in range(EPOCHS):
        state, info = env.reset()
        env.render()
        done = False
        rewards = []
        while not done:
            action, _states = model.predict(state)
            next_state, reward, done, trunc, info = env.step(action)
            env.render()
            rewards.append(reward)
            if done:
                cumulative_reward = sum(rewards)
                print("Reward: ", cumulative_reward, "\n")
                cumulative_rewards.append(cumulative_reward)
            state = next_state
    res_dict = {
        "iter": np.arange(len(cumulative_rewards)),
        "reward": np.array(cumulative_rewards)
    }
    
    res_df = pd.DataFrame(res_dict, columns=["iter", "reward"])
    res_df.to_csv("SAC_test_results.csv", index=False)

    env._release_flexsim()
    input("Waiting for input to close FlexSim...")
    env.close()


if __name__ == "__main__":
    main()