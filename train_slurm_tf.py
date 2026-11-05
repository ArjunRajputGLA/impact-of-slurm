import os
import json
import time
import socket
import argparse
import numpy as np

# Ensure compatibility with legacy Keras workflows under tf.distribute
os.environ["TF_USE_LEGACY_KERAS"] = "1"

parser = argparse.ArgumentParser(description="Distributed TensorFlow with Slurm & oneDNN")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--steps_per_epoch", type=int, default=15)
parser.add_argument("--per_worker_batch", type=int, default=64)
parser.add_argument("--omp_threads", type=int, default=1)
parser.add_argument("--onednn", type=str, default="1")
parser.add_argument("--optimize_pipeline", type=int, default=1)
args = parser.parse_args()

# 1. Hardware-Aware Thread Pinning & oneDNN Configuration
os.environ["TF_ENABLE_ONEDNN_OPTS"] = args.onednn
os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(args.omp_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "1"

import tensorflow as tf

# 2. Extract Slurm Environment Variables
slurm_rank = int(os.environ.get("SLURM_PROCID", 0))
slurm_world_size = int(os.environ.get("SLURM_NTASKS", 1))

if slurm_world_size > 1:
    if "SLURM_TF_ENDPOINTS" in os.environ and os.environ["SLURM_TF_ENDPOINTS"].strip():
        worker_endpoints = os.environ["SLURM_TF_ENDPOINTS"].split(",")
    else:
        def _alloc_port():
            with socket.socket() as s:
                s.bind(("", 0))
                return s.getsockname()[1]
        worker_endpoints = [f"localhost:{_alloc_port()}" for _ in range(slurm_world_size)]

    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {"worker": worker_endpoints},
        "task": {"type": "worker", "index": slurm_rank}
    })
else:
    # Standalone mode: clear TF_CONFIG to avoid unnecessary socket binding
    os.environ.pop("TF_CONFIG", None)

# 3. Setup Strategy (Ring-AllReduce across all cluster workers)
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# 4. Prepare Sharded Dataset using distribute_datasets_from_function
global_batch_size = args.per_worker_batch * slurm_world_size
num_samples = 2560
rng = np.random.default_rng(seed=42)
x_data = rng.standard_normal((num_samples, 32, 32, 3), dtype=np.float32)
y_data = rng.integers(0, 10, size=(num_samples,), dtype=np.int32)

def dataset_fn(input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    d = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    # Shard dataset cleanly per worker rank
    d = d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    if args.optimize_pipeline == 1:
        d = d.batch(batch_size).cache().repeat().prefetch(tf.data.AUTOTUNE)
    else:
        d = d.batch(batch_size).repeat()
    return d

dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)

# 5. Build Model & Optimizer within Strategy Scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)
    ])
    model.build(input_shape=(None, 32, 32, 3))

    # Linear Learning Rate Scaling based on cluster size
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001 * slurm_world_size)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )

def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

# 6. Custom Distributed Training Step (Explicit Ring-AllReduce via strategy.run)
# This completely eliminates Keras 3 model.fit() 'PerReplica' conversion errors
def train_step(inputs):
    x_batch, y_batch = inputs
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss = compute_loss(y_batch, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def distributed_train_step(dist_inputs):
    per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# 7. Execute Distributed Training Loop
dataset_iter = iter(dist_dataset)

# Warmup steps
for _ in range(3):
    distributed_train_step(next(dataset_iter))

# Timed Benchmark Loop
t0 = time.perf_counter()
last_loss = 0.0
for epoch in range(args.epochs):
    epoch_loss = 0.0
    for step in range(args.steps_per_epoch):
        batch = next(dataset_iter)
        step_loss = distributed_train_step(batch)
        epoch_loss += float(step_loss)
    last_loss = epoch_loss / max(args.steps_per_epoch, 1)

elapsed = time.perf_counter() - t0

total_images = args.epochs * args.steps_per_epoch * global_batch_size
throughput = total_images / elapsed

if slurm_rank == 0:
    print(f"[Slurm Rank 0 Summary] Workers: {slurm_world_size} | Per-Worker Batch: {args.per_worker_batch} | Global Batch: {global_batch_size} | OMP_THREADS: {args.omp_threads} | oneDNN: {args.onednn}")
    print(f"[Slurm Rank 0 Metrics] Training Time: {elapsed:.2f}s | Cluster Throughput: {throughput:.1f} samples/sec | Final Loss: {last_loss:.4f}")