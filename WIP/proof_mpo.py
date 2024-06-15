import torch
import torch.nn as nn

n = nn.Parameter(torch.ones(1) * 100)
a = nn.Parameter()
e = 0.2

optim = torch.optim.Adam(params=[n], lr=1.0)

# example_pi = torch.tensor([0.6, 0.4]).float()

example_pi = nn.Sequential(
    nn.Linear(2, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 2),
    nn.Softmax(dim=-1),
)

example_Q = torch.randint(low=-5, high=10, size=[3, 2]).float()

# print(example_Q)

# print(example_pi(example_Q))

# sys.exit()


def expectation(Q, policy):
    return torch.sum(Q * policy, dim=-1, keepdim=True)


# print(f"Expectation of random policy: {expectation(example_Q, example_pi(example_Q)).squeeze()}")
# print(f"Expectation of q_ij policy: {expectation(example_Q, torch.softmax(example_Q/n, -1)).squeeze()}")
n = 10
for i in range(50):
    # n.zero_grad()

    def dual(n):
        avg_over_actions = torch.exp(example_Q / n)
        # print(example_Q)
        # print(avg_over_actions)
        avg_over_actions = avg_over_actions.mean(dim=-1)
        # print(avg_over_actions)

        avg_over_states = avg_over_actions.log()
        # print(avg_over_states)
        avg_over_states = avg_over_states.mean(dim=-1)
        # print(avg_over_states)
        loss = n * e + n * avg_over_states

        return nn.functional.softplus(loss)

    loss = dual(n=n)
    print(f"{i}: {loss.item(), n}")

    n = loss  # *0.1

    continue

    # print(f'{i}: {loss.item(), n.item(), act_weights, expectation(example_Q, act_weights)}')
    loss.backward()
    optim.step()

    n = loss

print(
    f"Expectation of random policy: {expectation(example_Q, example_pi(example_Q)).squeeze()}"
)
print(
    f"Expectation of q_ij policy: {expectation(example_Q, torch.softmax(example_Q/n, -1)).squeeze()}"
)
