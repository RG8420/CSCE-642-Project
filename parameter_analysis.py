import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import gymnasium as gym
import itertools
import csv
import os
import time
from flexsim_env_v2025 import FlexSimEnv
from stable_baselines3.common.env_checker import check_env

# =======================================================
# 1. Actor-Critic Network Definition (Unchanged)
# =======================================================

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dims):
        super().__init__()

        self.action_dims = action_dims
        self.num_actions = len(action_dims)

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )

        self.pi = nn.Linear(64, int(np.sum(action_dims)))
        self.v  = nn.Linear(64, 1)

        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, np.sqrt(2))

        orthogonal_init(self.pi, 0.01)
        orthogonal_init(self.v, 1.0)

    def _split_logits(self, logits):
        splits = []
        idx = 0
        for n in self.action_dims:
            splits.append(logits[:, idx:idx+n])
            idx += n
        return splits

    def forward(self, x):
        x = self.shared(x)
        logits = self.pi(x)
        value = self.v(x)
        return logits, value

    def act(self, obs):
        with torch.no_grad():
            # obs is expected to be a 1D tensor here, use unsqueeze(0) for batch dim
            logits, value = self.forward(obs.unsqueeze(0))
            splits = self._split_logits(logits)
            dists = [torch.distributions.Categorical(logits=s) for s in splits]

            actions = torch.tensor([d.sample() for d in dists])
            logp = torch.stack([d.log_prob(a) for d,a in zip(dists, actions)]).sum()

        return actions.numpy().tolist(), logp, value.squeeze()


# =======================================================
# 2. Advantage Computation (Unchanged)
# =======================================================

def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advs = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advs[t] = gae
    returns = advs + values[:-1]
    return advs, returns

# =======================================================
# 3. PPO Agent Class (Unchanged core logic)
# =======================================================

class PPO:
    def __init__(self, obs_dim, action_dims,
             lr=3e-4, gamma=0.99, lam=0.95, clip_range=0.2,
             rollout_size=512, minibatch_size=64, n_epochs=4,
             max_grad_norm=0.5, ent_coef=0.01, vf_coef=0.5):

        self.gamma = gamma
        self.lam = lam
        self.clip = clip_range
        self.rollout_size = rollout_size
        self.minibatch_size = minibatch_size
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        self.policy = ActorCritic(obs_dim, action_dims)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.values = []
        self.ep_rewards = []
        self.ep_lengths = []
        # Store last update metrics for logging/analysis
        self.last_metrics = {}

    def clear_buffer(self):
        # Only clears rollout data, NOT episode stats
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.logprobs.clear()
        self.values.clear()

    def learn(self, env, total_timesteps, render=False):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)

        timestep = 0
        ep_reward = 0
        ep_length = 0

        while timestep < total_timesteps:

            self.clear_buffer()

            # Rollout phase
            for _ in range(self.rollout_size):

                action, logp, value = self.policy.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # store transition
                self.states.append(obs.numpy())
                self.actions.append(action)
                self.rewards.append(reward)
                self.dones.append(done)
                self.logprobs.append(logp)
                self.values.append(value.item())

                # episode stats
                ep_reward += reward
                ep_length += 1

                obs = torch.tensor(next_obs, dtype=torch.float32)
                timestep += 1

                if done:
                    self.ep_rewards.append(ep_reward)
                    self.ep_lengths.append(ep_length)

                    ep_reward = 0
                    ep_length = 0

                    obs, _ = env.reset()
                    obs = torch.tensor(obs, dtype=torch.float32)

                if timestep >= total_timesteps:
                    break

            # Last value for GAE
            with torch.no_grad():
                # obs is already 1D tensor, need to unsqueeze for forward pass
                _, last_value = self.policy.forward(obs.unsqueeze(0)) 
                last_value = last_value.item()

            # Update PPO
            self.update(last_value)

            # print(f"Total timesteps so far: {timestep}") # Disable print for analysis

        return self

    def update(self, last_value):

        values = np.array([v for v in self.values] + [last_value])
        rewards = np.array(self.rewards)
        dones = np.array(self.dones, dtype=np.float32)

        advs, returns = compute_gae(rewards, values, dones,
                                    self.gamma, self.lam)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        states = torch.tensor(self.states, dtype=torch.float32)
        actions = torch.tensor(self.actions)
        old_logps = torch.stack(self.logprobs).detach()
        advs = torch.tensor(advs, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        dataset_size = len(states)

        last_approx_kl = 0
        last_clip_fraction = 0
        last_policy_loss = 0
        last_value_loss = 0
        last_entropy = 0

        # PPO optimization
        for _ in range(self.n_epochs):
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, self.minibatch_size):
                idx = indices[start:start+self.minibatch_size]

                # === Create MultiCategorical distributions ===
                logits, new_values = self.policy(states[idx])
                splits = self.policy._split_logits(logits)
                dists = [torch.distributions.Categorical(logits=s) for s in splits]

                new_logps = torch.stack([
                    d.log_prob(a) for d, a in zip(dists, actions[idx].T)
                ]).sum(dim=0)

                entropy = torch.stack([d.entropy() for d in dists]).mean()

                ratio = torch.exp(new_logps - old_logps[idx])

                pg1 = advs[idx] * ratio
                pg2 = advs[idx] * torch.clamp(
                    ratio,
                    1 - self.clip,
                    1 + self.clip)

                policy_loss = -torch.min(pg1, pg2).mean()
                value_loss = (returns[idx] - new_values.squeeze()).pow(2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                last_policy_loss = policy_loss.item()
                last_value_loss = value_loss.item()
                last_entropy = entropy.item()
                last_approx_kl = (old_logps[idx] - new_logps).mean().item()
                last_clip_fraction = (
                    (ratio > (1 + self.clip)) | (ratio < (1 - self.clip))
                ).float().mean().item()

        # Update last metrics for logging
        self.last_metrics = {
            'policy_loss': last_policy_loss,
            'value_loss': last_value_loss,
            'entropy': last_entropy,
            'approx_kl': last_approx_kl,
            'clip_fraction': last_clip_fraction
        }

        # self.clear_buffer() # clear_buffer is called at the start of the next rollout

# =======================================================
# 4. Hyperparameter Analysis Logic
# =======================================================

# =======================================================
# 4. Hyperparameter Analysis Logic (FIXED FOR SOCKET ERROR)
# =======================================================

def run_parameter_analysis():
    # --- Hyperparameter Grid ---
    param_gamma = [0.91, 0.95, 0.97, 0.99]
    param_lam = [0.91, 0.95, 0.97, 0.99]
    param_vf_coef = [0.1, 0.5, 0.9]
    param_ent_coef = [0.001, 0.01, 0.1]
    
    # Other fixed parameters
    TOTAL_TIMESTEPS = 10000
    FIXED_LR = 3e-4
    FIXED_CLIP = 0.2
    FIXED_ROLLOUT = 512
    FIXED_MINIBATCH = 64
    FIXED_EPOCHS = 10
    
    # --- Robustness Settings ---
    MAX_RETRIES = 5  # Allow up to 5 attempts to start a single run
    RETRY_DELAY = 5  # Wait 5 seconds between retries to clear the port
    
    # --- File setup ---
    CSV_FILE = "hyperparameter_analysis_results.csv"
    
    # Prepare CSV file and write header
    fieldnames = ['gamma', 'lam', 'vf_coef', 'ent_coef', 'final_mean_reward', 'num_episodes', 'final_policy_loss', 'final_value_loss']
    with open(CSV_FILE, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # --- Iteration Loop ---
    combinations = list(itertools.product(param_gamma, param_lam, param_vf_coef, param_ent_coef))
    print(f"Total {len(combinations)} parameter combinations to test with {TOTAL_TIMESTEPS} timesteps each.")

    for i, (gamma, lam, vf_coef, ent_coef) in enumerate(combinations):
        current_run_successful = False
        
        # New: Retry loop for environment initialization failures
        for attempt in range(MAX_RETRIES):
            if current_run_successful:
                break
                
            print(f"\n--- Starting Run {i+1}/{len(combinations)} (Attempt {attempt+1}): gamma={gamma}, lam={lam}, vf_coef={vf_coef}, ent_coef={ent_coef} ---")
            
            env = None # Initialize env outside try block for cleanup
            
            try:
                # 1. Initialize Environment
                env = FlexSimEnv(
                    flexsimPath = r"D:\Softwares\flexsim2024\program\flexsim.exe",
                    modelPath = r"D:\Projects\CSCE642_DRL\Project\FJSSP_v2_15_Types_3_Machines_4_Operations_v7.fsm",
                    verbose = False,
                    visible = False
                )

                # Get dimensions (unchanged)
                obs_dim = env.observation_space.shape[0]
                if isinstance(env.action_space, gym.spaces.MultiDiscrete):
                    action_dims = np.array(env.action_space.nvec, dtype=int)
                elif isinstance(env.action_space, gym.spaces.Discrete):
                    action_dims = np.array([env.action_space.n], dtype=int)
                else:
                    raise NotImplementedError(f"Unsupported action space: {env.action_space}")

                # 2. Initialize Agent (unchanged)
                agent = PPO(
                    obs_dim, action_dims,
                    lr=FIXED_LR, gamma=gamma, lam=lam, clip_range=FIXED_CLIP,
                    rollout_size=FIXED_ROLLOUT, minibatch_size=FIXED_MINIBATCH, n_epochs=FIXED_EPOCHS,
                    max_grad_norm=0.5, ent_coef=ent_coef, vf_coef=vf_coef
                )

                # 3. Train the agent (unchanged)
                agent.learn(env, total_timesteps=TOTAL_TIMESTEPS, render=False)

                # 4. Extract Results (unchanged)
                final_mean_reward = np.mean(agent.ep_rewards[-50:]) if agent.ep_rewards else 0.0
                num_episodes = len(agent.ep_rewards)

                # 5. Store results for CSV (unchanged)
                results = {
                    'gamma': gamma, 'lam': lam, 'vf_coef': vf_coef, 'ent_coef': ent_coef,
                    'final_mean_reward': final_mean_reward, 'num_episodes': num_episodes,
                    'final_policy_loss': agent.last_metrics.get('policy_loss', np.nan),
                    'final_value_loss': agent.last_metrics.get('value_loss', np.nan)
                }
                
                print(f"Result: Mean Reward={final_mean_reward:.2f}, Episodes={num_episodes}")
                current_run_successful = True # Run was successful

            except OSError as e:
                if "10048" in str(e):
                    # Specific handler for the port binding error
                    print(f"--- Connection Error: [WinError 10048] Port is busy (Attempt {attempt+1}/{MAX_RETRIES}) ---")
                    if attempt < MAX_RETRIES - 1:
                        print(f"Waiting {RETRY_DELAY}s for port cleanup...")
                        time.sleep(RETRY_DELAY)
                    else:
                        print("Max retries reached. Recording failure.")
                        # Record failure with NaN results
                        results = {
                            'gamma': gamma, 'lam': lam, 'vf_coef': vf_coef, 'ent_coef': ent_coef,
                            'final_mean_reward': np.nan, 'num_episodes': np.nan,
                            'final_policy_loss': np.nan, 'final_value_loss': np.nan
                        }
                        current_run_successful = True # Log failure and move on
                else:
                    # Handle other OS errors
                    print(f"--- Unexpected OSError in run: {e} ---")
                    current_run_successful = True # Treat as final attempt and log failure
                    results = { # Record failure with NaN results
                        'gamma': gamma, 'lam': lam, 'vf_coef': vf_coef, 'ent_coef': ent_coef,
                        'final_mean_reward': np.nan, 'num_episodes': np.nan,
                        'final_policy_loss': np.nan, 'final_value_loss': np.nan
                    }

            except Exception as e:
                # Handle general training/FlexSim communication errors
                print(f"--- General ERROR in run: {e} ---")
                results = { # Record failure with NaN results
                    'gamma': gamma, 'lam': lam, 'vf_coef': vf_coef, 'ent_coef': ent_coef,
                    'final_mean_reward': np.nan, 'num_episodes': np.nan,
                    'final_policy_loss': np.nan, 'final_value_loss': np.nan
                }
                current_run_successful = True # Treat as final attempt and log failure

            finally:
                # 7. Release FlexSim resource and close env after every attempt
                if env is not None:
                    try:
                        env._release_flexsim()
                        env.close()
                        time.sleep(1) # Extra short delay after successful cleanup
                    except Exception:
                        pass # Ignore errors if FlexSim failed to launch/close properly

        # 6. Write results to CSV (only if a successful run or max retries reached)
        if current_run_successful:
            with open(CSV_FILE, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(results)

    print(f"\n✅ Hyperparameter analysis complete. Results saved to {CSV_FILE}")
    
# =======================================================
# 5. Execution
# =======================================================

if __name__ == "__main__":
    run_parameter_analysis()
    