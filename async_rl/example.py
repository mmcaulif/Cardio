# type: ignore

import time
from collections import deque

import ray

# Initialize Ray
ray.init()


# Central class that will collect and print the number of states
@ray.remote
class StateCollector:
    def __init__(self):
        self.states = deque()

    # Method to collect states from workers
    def collect_state(self, state):
        self.states.append(state)

    # Method to count the number of states collected
    def count_states(self):
        return len(self.states)

    # Method to reset the state counter
    def reset(self):
        self.states.clear()


# Worker class that simulates state collection
@ray.remote
class Worker:
    def __init__(self, state_collector):
        self.state_collector = state_collector

    # Method to simulate generating and sending states
    def collect_states(self):
        for _ in range(100):  # Simulate 100 state collections
            state = {"observation": _, "reward": 1.0}  # Example state
            ray.get(self.state_collector.collect_state.remote(state))
            time.sleep(0.1)  # Simulate time between collecting states


# Function to print the state count every second
def monitor_state_collection(state_collector):
    try:
        while True:
            time.sleep(1)
            count = ray.get(state_collector.count_states.remote())
            print(f"States received so far: {count}")
    except KeyboardInterrupt:
        print("Monitoring stopped.")


# Create the StateCollector instance
state_collector = StateCollector.remote()

# Create multiple workers
num_workers = 4
workers = [Worker.remote(state_collector) for _ in range(num_workers)]

# Start state collection by workers
for worker in workers:
    worker.collect_states.remote()

# Monitor the collection process
monitor_state_collection(state_collector)
