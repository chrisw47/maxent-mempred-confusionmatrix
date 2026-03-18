#!/usr/bin/env python
"""
Diagnostic script to check if JAX is using GPU and where bottlenecks are.
Run in Colab with: !python check_gpu_usage.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from physf import KL, generate_binary_states, P_data

print("=" * 60)
print("JAX GPU/Device Check")
print("=" * 60)
print(f"Available devices: {jax.devices()}")
print(f"Devices: {[str(d) for d in jax.devices()]}")

# Create test data
np.random.seed(42)
k = 5
n_samples = 10000
test_data = np.random.randint(0, 2, size=(n_samples, k))
state_arr = generate_binary_states(k)
data_probs = P_data(test_data)

upper_indices = np.triu_indices(k)
theta0 = np.zeros((k, k))
theta0[upper_indices] = np.random.normal(0, 0.3, len(upper_indices[0]))
theta0 = theta0 + theta0.T
theta0_flat = theta0.flatten()

print("\n" + "=" * 60)
print("Timing KL function calls")
print("=" * 60)

# Warm up
_ = KL(theta0_flat, test_data, data_probs, state_arr)

# Time 100 calls
start = time.time()
for _ in range(100):
    kl_val = KL(theta0_flat, test_data, data_probs, state_arr)
elapsed = time.time() - start

print(f"100 KL calls: {elapsed:.3f}s ({elapsed/100*1000:.2f} ms per call)")

# Test actual GPU tensor operations
print("\n" + "=" * 60)
print("GPU tensor operation test")
print("=" * 60)

# Large matrix multiplication (should use GPU)
A = jnp.ones((5000, 5000))
B = jnp.ones((5000, 5000))

start = time.time()
for _ in range(10):
    C = jnp.matmul(A, B)
    _ = C.block_until_ready()  # Ensure computation completes
gpu_time = time.time() - start

print(f"10x 5000x5000 matmul on {jax.devices()[0]}: {gpu_time:.3f}s")

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)
if 'cpu' in str(jax.devices()[0]).lower():
    print("⚠️  JAX is using CPU, not GPU!")
    print("   Check: Runtime → Change runtime type → GPU")
else:
    print("✓ JAX is using GPU/TPU")
    print("  If KL function is slow, the bottleneck may be:")
    print("  1. Data transfer overhead (NumPy ↔ JAX)")
    print("  2. Small operation size (overhead > benefit)")
    print("  3. I/O operations (data loading from Drive)")
