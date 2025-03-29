import torch
import time
import sys

def benchmark_fp16_mps(matrix_size=4096, iterations=100):
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this system.")
    if not torch.backends.mps.is_built():
        raise RuntimeError("PyTorch is not built with MPS support.")

    device = torch.device("mps")
    dtype = torch.float16

    print(f"Benchmarking FP16 matmul on MPS ({matrix_size}x{matrix_size}) for {iterations} iterations...")

    # Create two large FP16 matrices on MPS
    try:
        a = torch.randn((matrix_size, matrix_size), dtype=dtype, device=device)
        b = torch.randn((matrix_size, matrix_size), dtype=dtype, device=device)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to allocate tensors on MPS: {e}")

    # Warm-up
    for _ in range(10):
        torch.matmul(a, b)
    torch.mps.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        c = torch.matmul(a, b)
    torch.mps.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    time_per_iter = total_time / iterations
    gflops = 2 * (matrix_size ** 3) * iterations / (total_time * 1e9)

    print(f"Total time: {total_time:.4f} sec")
    print(f"Average time per iteration: {time_per_iter:.6f} sec")
    print(f"Estimated throughput: {gflops:.2f} GFLOPS (FP16)")

if __name__ == "__main__":
    try:
        benchmark_fp16_mps()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
