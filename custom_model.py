import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from flexsim_env_v2025 import FlexSimEnv
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import numpy as np

# Actor-Critic Network (with orthogonal initialization + correct policy head)
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dims):   # <-- changed
        super().__init__()

        self.action_dims = action_dims          # store nvec (e.g. [4,3,5])
        self.num_actions = len(action_dims)     # number of sub-actions

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )

        # actor head: outputs sum(nvec) logits
        self.pi = nn.Linear(64, int(np.sum(action_dims)))

        # critic head (unchanged)
        self.v  = nn.Linear(64, 1)

        # === Orthogonal init same as before ===
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, np.sqrt(2))

        orthogonal_init(self.pi, 0.01)
        orthogonal_init(self.v, 1.0)

    def _split_logits(self, logits):
        """Split flat logits into each discrete dimension."""
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
            logits, value = self.forward(obs.unsqueeze(0))  # add batch
            splits = self._split_logits(logits)
            dists = [torch.distributions.Categorical(logits=s) for s in splits]

            actions = torch.tensor([d.sample() for d in dists])
            logp = torch.stack([d.log_prob(a) for d,a in zip(dists, actions)]).sum()

        return actions.numpy().tolist(), logp, value.squeeze()


# GAE Advantage Computation (correct SB3 implementation)
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

# PPO Agent (SB3-consistent training loop)
class PPO:
    def __init__(self, obs_dim, action_dims,
             lr=3e-4,
             gamma=0.99,
             lam=0.95,
             clip_range=0.2,
             rollout_size=512,
             minibatch_size=64,
             n_epochs=4,
             max_grad_norm=0.5,
             ent_coef=0.01,
             vf_coef=0.5):

        self.gamma = gamma
        self.lam = lam
        self.clip = clip_range
        self.rollout_size = rollout_size
        self.minibatch_size = minibatch_size
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        # === Instantiate policy with MultiDiscrete support ===
        self.policy = ActorCritic(obs_dim, action_dims)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # rollout buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.values = []

        # episode stats
        self.ep_rewards = []
        self.ep_lengths = []


    # ---------------------------------------------------------
    # Only clears rollout data, NOT episode stats
    # ---------------------------------------------------------
    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.logprobs.clear()
        self.values.clear()

    # ---------------------------------------------------------
    # Training Loop with working reward tracking
    # ---------------------------------------------------------
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
                    # Store the completed episode stats
                    self.ep_rewards.append(ep_reward)
                    self.ep_lengths.append(ep_length)

                    # Reset counters
                    ep_reward = 0
                    ep_length = 0

                    obs, _ = env.reset()
                    obs = torch.tensor(obs, dtype=torch.float32)

                if timestep >= total_timesteps:
                    break

            # Last value for GAE
            with torch.no_grad():
                _, last_value = self.policy.forward(obs)
                last_value = last_value.item()

            # Update PPO
            self.update(last_value)

            print(f"Total timesteps so far: {timestep}")

        print("Training complete.")
        return self

    # ---------------------------------------------------------
    # PPO Update Step (unchanged except logging)
    # ---------------------------------------------------------
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
                splits = self.policy._split_logits(logits)  # split logits into action dims

                # create a categorical dist for each group of logits
                dists = [torch.distributions.Categorical(logits=s) for s in splits]

                # NOTE: actions are (batch, dims). We must transpose for iterating dims
                # compute log-prob for each action dimension, then sum
                new_logps = torch.stack([
                    d.log_prob(a) for d, a in zip(dists, actions[idx].T)
                ]).sum(dim=0)

                # compute entropy across dimensions, mean across batch
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

        # -----------------------------------------------------
        # SAFE logging: avoids nan if no episode finished yet
        # -----------------------------------------------------
        if len(self.ep_rewards) > 0:
            ep_rew_mean = np.mean(self.ep_rewards[-10:])   # last 10 episodes
        else:
            ep_rew_mean = 0.0

        print("--------------------------------")
        print(f"ep_rew_mean     | {ep_rew_mean:.2f}")
        print(f"policy_loss     | {last_policy_loss:.3f}")
        print(f"value_loss      | {last_value_loss:.3f}")
        print(f"entropy         | {last_entropy:.3f}")
        print(f"approx_kl       | {last_approx_kl:.3e}")
        print(f"clip_fraction   | {last_clip_fraction:.2f}")
        print("--------------------------------")

    
        self.clear_buffer()


env = FlexSimEnv(
    flexsimPath = r"D:\Softwares\flexsim2024\program\flexsim.exe",
    modelPath = r"D:\Projects\CSCE642_DRL\Project\FJSSP_v2_15_Types_3_Machines_4_Operations_v7.fsm",
    verbose = False,
    visible = False
)

# Check environment
check_env(env, warn=True)

# observation dimension
obs_dim = env.observation_space.shape[0]

# --- IMPORTANT: build action_dims as a 1D array/list ---
if isinstance(env.action_space, gym.spaces.MultiDiscrete):
    action_dims = np.array(env.action_space.nvec, dtype=int)   # e.g. [4,3,5]
elif isinstance(env.action_space, gym.spaces.Discrete):
    action_dims = np.array([env.action_space.n], dtype=int)    # wrap single n
else:
    raise NotImplementedError(f"Unsupported action space: {env.action_space}")


agent = PPO(
    obs_dim,
    action_dims,
    lr=3e-4,
    gamma=0.95,
    lam=0.95,
    clip_range=0.2,
    rollout_size=512,
    minibatch_size=64,
    n_epochs=100,
    max_grad_norm=0.5,
    ent_coef=0.01,
    vf_coef=0.5
)

# -------------------------------------------------------
# Safe Training (FlexSim always closes)
# -------------------------------------------------------
agent.learn(env, total_timesteps=10000, render=False)

# Save Model
torch.save(agent.policy.state_dict(), "Custom_PPO_Thesis_Model.pth")
print("\nModel saved as Custom_PPO_Thesis_Model.pth")

env._release_flexsim()
env.close()
print("FlexSim released and environment closed.")