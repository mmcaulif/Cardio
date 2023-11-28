import envpool
import numpy as np
import time
from tqdm import trange

# Start timer
start_time = time.time()

env = envpool.make_gymnasium("Hopper-v4")

obs = env.reset()

for _ in trange(1_000_000):
    act = np.array([env.action_space.sample()])  
    # act = env.action_space.sample()

    obs, rew, term, trunc, info = env.step(act)
    done = term or trunc
    if done:
        obs = env.reset()

# End timer
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time) 