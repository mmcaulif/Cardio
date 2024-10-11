# type: ignore

import time

import ray

# Initialize Ray
ray.init()


# Define the first actor class
@ray.remote
class ActorA:
    def __init__(self):
        self.messages = []
        self.stop = False

    # Method to send a message to ActorB
    async def send_message(self, actor_b, message):
        while not self.stop:
            print(f"ActorA sending message: {message}")
            await actor_b.receive_message.remote(message)
            await self.process_messages()
            time.sleep(1)  # Simulate some delay between messages

    # Method to receive a message from ActorB
    async def receive_message(self, message):
        print(f"ActorA received message: {message}")
        self.messages.append(message)

    # Process received messages (can add logic here)
    async def process_messages(self):
        if self.messages:
            print(f"ActorA is processing: {self.messages}")
            self.messages = []  # Clear messages after processing

    # Stop the continuous loop
    def stop_sending(self):
        self.stop = True


# Define the second actor class
@ray.remote
class ActorB:
    def __init__(self):
        self.messages = []
        self.stop = False

    # Method to send a message to ActorA
    async def send_message(self, actor_a, message):
        while not self.stop:
            print(f"ActorB sending message: {message}")
            await actor_a.receive_message.remote(message)
            await self.process_messages()
            time.sleep(1)  # Simulate some delay between messages

    # Method to receive a message from ActorA
    async def receive_message(self, message):
        print(f"ActorB received message: {message}")
        self.messages.append(message)

    # Process received messages (can add logic here)
    async def process_messages(self):
        if self.messages:
            print(f"ActorB is processing: {self.messages}")
            self.messages = []  # Clear messages after processing

    # Stop the continuous loop
    def stop_sending(self):
        self.stop = True


# Create instances of both actors
actor_a = ActorA.remote()
actor_b = ActorB.remote()

# Start continuous message passing between ActorA and ActorB
# This will keep both actors sending and receiving messages in an infinite loop
actor_a_task = actor_a.send_message.remote(actor_b, "Message from ActorA")
actor_b_task = actor_b.send_message.remote(actor_a, "Message from ActorB")

# Let it run for a few seconds, then stop
time.sleep(10)

# Stop both actors from sending messages
ray.get([actor_a.stop_sending.remote(), actor_b.stop_sending.remote()])

# Shutdown Ray after use
ray.shutdown()
