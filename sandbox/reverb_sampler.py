import time

import jax
import reverb

dataset = reverb.TrajectoryDataset.from_table_signature(
    server_address="localhost:50827",   # Needs to be manually set and varies form run to run
    table="table",
    max_in_flight_samples_per_worker=(2 * 32 // jax.process_count()),
)
dataset = dataset.batch(32, drop_remainder=True)
dataset = dataset.as_numpy_iterator()

while True:
    time.sleep(0.5)
    print(next(dataset))
