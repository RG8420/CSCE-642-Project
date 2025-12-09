import os
import csv
from flexsim_env_v2025 import FlexSimEnv 
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO # Only PPO is needed for this revision
# from stable_baselines3 import SAC, TD3, A2C, DQN # Commented out unused imports
# from stable_baselines3.common.env_util import make_vec_env # Commented out unused import

# --- Configuration Constants for Results ---
RESULTS_DIR = "PPO_Test_Results"
CSV_FILENAME = os.path.join(RESULTS_DIR, "PPO_Test_Run_Log.csv")
TEST_EPOCHS = 100 # Total number of test episodes

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

    # Training a baselines3 PPO model in the environment
    # Note: verbose=1 here will print training progress to the console
    model = PPO("MlpPolicy", env, verbose=1) 
    print("Training model...")
    model.learn(total_timesteps=10000)
    
    # save the model
    print("Saving model...")
    model.save("MyTrainedModel_ThesisRLModel_Breakdown")

    input("Waiting for input to do some test runs...")
    
    # --- Setup CSV Logging ---
    # Create the directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Open the CSV file and write the header
    with open(CSV_FILENAME, 'w', newline='') as csvfile:
        fieldnames = ['Episode', 'Cumulative_Reward']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Starting {TEST_EPOCHS} test runs and logging results to {CSV_FILENAME}...")
        
        # Run test episodes using the trained model
        for i in range(TEST_EPOCHS):
            # Use deterministic=True for evaluation runs
            state, info = env.reset()
            # env.render() # Rendering is usually slow, keep commented unless debugging
            done = False
            rewards = []
            
            # The testing loop
            while not done:
                # Use deterministic=True for trained model evaluation
                action, _states = model.predict(state, deterministic=True) 
                next_state, reward, done, trunc, info = env.step(action)
                # env.render()
                rewards.append(reward)
                state = next_state
            
            # End of episode
            cumulative_reward = sum(rewards)
            print(f"Episode {i+1}/{TEST_EPOCHS}: Reward = {cumulative_reward}")
            
            # Write the result to the CSV file
            writer.writerow({
                'Episode': i + 1, 
                'Cumulative_Reward': cumulative_reward
            })

    print(f"\nAll test results saved to: **{CSV_FILENAME}**")
    
    # Final environment cleanup
    env._release_flexsim()
    input("Waiting for input to close FlexSim...")
    env.close()


if __name__ == "__main__":
    main()