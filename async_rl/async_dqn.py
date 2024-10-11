# type: ignore

import time

import gymnasium as gym
import ray

# Initialize Ray
ray.init()


# Define the first actor class
@ray.remote
class Learner:
    def __init__(self):
        self.states = []
        self.stop = False

    def run(self):
        while not self.stop:
            self.process_message()
            time.sleep(1)  # Simulate some delay between messages

    def process_message(self):
        print(len(self.states))

    # Method to receive a message from ActorB
    def receive_message(self, state):
        # print(f"ActorA received message: {message}")
        self.states.append(state)

    # Stop the continuous loop
    def stop_sending(self):
        self.stop = True


# Define the second actor class
@ray.remote
class Worker:
    def __init__(self, learner):
        self.learner = learner
        self.env = gym.make("CartPole-v1")
        self.messages = []
        self.stop = False

    # Method to send a message to ActorA
    def run(self):
        s, _ = self.env.reset()
        while not self.stop:
            a = self.env.action_space.sample()
            s, r, d, t, _ = self.env.step(a)
            if d or t:
                s, _ = self.env.reset()

            ray.get(self.learner.receive_message.remote(s))

    # Stop the continuous loop
    def stop_sending(self):
        self.stop = True


# Create instances of both actors
learner = Learner.remote()

workers = [Worker.remote(learner) for _ in range(8)]

# Start continuous message passing between ActorA and ActorB
# This will keep both actors sending and receiving messages in an infinite loop

for worker in workers:
    worker.run.remote()

try:
    while True:
        time.sleep(1)
        ray.get(learner.process_message.remote())
except KeyboardInterrupt:
    print("Monitoring stopped.")


# Stop both actors from sending messages
# ray.get([worker.stop_sending.remote(), learner.stop_sending.remote()])

# Shutdown Ray after use
ray.shutdown()
