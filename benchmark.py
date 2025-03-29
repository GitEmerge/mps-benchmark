import torch
import time
import sys
import platform
import os
import gc

CHECKSUM_PATH = ".last_checksum.txt"

def benchmark_fp16_mps(matrix_size=2048, iterations=100):
    print("========== PyTorch MPS FP16 Benchmark ==========")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"Torch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    if not torch.backends.mps.is_available():
        raise RuntimeError("❌ MPS is not available on this system.")
    if not torch.backends.mps.is_built():
        raise RuntimeError("❌ PyTorch is not built with MPS support.")

    device = torch.device("mps")
    dtype = torch.float16

    print(f"\n✅ Running benchmark on device: {device}")
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"Iterations: {iterations}")
    print(f"Data type: {dtype}")
    print("===============================================\n")

    print("📦 Allocating input tensors on MPS...")
    try:
        a = torch.randn((matrix_size, matrix_size), dtype=dtype, device=device)
        b = torch.randn((matrix_size, matrix_size), dtype=dtype, device=device)
    except RuntimeError as e:
        raise RuntimeError(f"❌ Failed to allocate tensors on MPS: {e}")
    print("✅ Tensors allocated successfully.\n")

    print("🔥 Warming up (10 iterations)...")
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.mps.synchronize()
    print("✅ Warm-up complete.\n")

    # 🧹 Clean up memory after warm-up
    torch.mps.empty_cache()
    gc.collect()

    print("⏱️ Starting benchmark...")
    torch.mps.synchronize()
    start_time = time.time()

    checksum = 0.0
    for i in range(iterations):
        if i % 10 == 0:
            print(f"➡️  Iteration {i+1}/{iterations}...")
        try:
            c = torch.matmul(a, b)
            checksum += torch.sum(c.float()).item()  # Convert to float32 to avoid inf
        except RuntimeError as e:
            print(f"❌ Error during iteration {i+1}: {e}", file=sys.stderr)
            raise e
        finally:
            del c
            torch.mps.empty_cache()
            gc.collect()

    torch.mps.synchronize()
    end_time = time.time()

    if not torch.isfinite(torch.tensor(checksum)):
        raise RuntimeError("❌ Checksum is not finite (inf or NaN). Accumulation may have overflowed.")

    total_time = end_time - start_time
    time_per_iter = total_time / iterations
    gflops = 2 * (matrix_size ** 3) * iterations / (total_time * 1e9)

    print("\n📊 Benchmark results:")
    print(f"🕒 Total time: {total_time:.4f} seconds")
    print(f"⚡ Average time per iteration: {time_per_iter:.6f} seconds")
    print(f"🚀 Estimated throughput: {gflops:.2f} GFLOPS (FP16)")
    print(f"🧮 Sanity check (sum of all c values): {checksum:.4f}")
    print("✅ Benchmark completed successfully.")
    print("===============================================\n")

    with open(CHECKSUM_PATH, "w") as f:
        f.write(f"{checksum:.4f}\n")

if __name__ == "__main__":
    try:
        # Optional: disable memory cap on MPS
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        benchmark_fp16_mps()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
