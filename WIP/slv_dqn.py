import jax
import cardio_rl as crl
import gymnasium as gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F

"""
Stochastic latent variables for computing intrinsic rewards in deep RL

- Try this again with DDPG in https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/
"""

class Q_critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.mu = nn.Linear(64, 64)
        self.logvar = nn.Linear(64, 64)
        self.l2 = nn.Linear(64, action_dim)

    def forward(self, state, aux = False):
        x = F.relu(self.l1(state))
        mu = self.mu(x)
        log_var = self.logvar(mu)
        std = th.exp(0.5 * log_var)
        eps = th.rand_like(std)
        z = mu + eps * std
        q = self.l2(z)
        if aux:
            return q, z, mu, log_var
        return q


class DQN(crl.Agent):
    def __init__(self, intrinsic_coeff: float = 5e-3, beta: float = 0.5):
        self.intrinsic_coeff = intrinsic_coeff  # Coeff for turning kl divergence to intrinsic reward
        self.beta = beta    # Coeff for elbo loss

        self.critic = Q_critic(2, 3)
        self.targ_critic = Q_critic(2, 3)
        self.targ_critic.load_state_dict(self.critic.state_dict())
        self.update_count = 0
        self.optimizer = th.optim.Adam(self.critic.parameters(), lr=1e-4)

    def update(self, batches):
        for data in batches:
            data = jax.tree.map(crl.utils.to_torch, data)
            s, a, r, s_p, d = data["s"], data["a"], data["r"], data["s_p"], data["d"]

            q, z, mu, logvar = self.critic(s, aux=True)
            kl_div = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            r_i = (self.intrinsic_coeff * kl_div).unsqueeze(-1) # Detach not needed as y is already detached

            q = q.gather(-1, a.long())

            a_p = self.critic(s_p).argmax(-1, keepdim=True)
            q_p = self.targ_critic(s_p).gather(-1, a_p.long())
            y = (r + r_i) + 0.99 * q_p * (1 - d)

            kl_loss = th.mean(r_i)
            loss = F.mse_loss(q, y.detach()) + self.beta * kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_count += 1
            if self.update_count % 1_000 == 0:
                self.targ_critic.load_state_dict(self.critic.state_dict())

    def _step(self, state):
        th_state = th.from_numpy(state).unsqueeze(0).float()
        action = self.critic(th_state).argmax().detach().numpy()
        return action

    def step(self, state):
        action = self._step(state)
        return action, {}
    
    def eval_step(self, state):
        return self._step(state), {}


def main():
    runner = crl.OffPolicyRunner(
        env=gym.make("MountainCar-v0"),
        agent=DQN(),
        rollout_len=5,
        batch_size=32,
    )
    runner.run(100_000, eval_interval=10_000, eval_episodes=1)


if __name__ == "__main__":
    main()
