
import csv
import statistics
import subprocess
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS = 20
WARMUP_RUNS = 3

TESTS = [
    ("CPU", "CPUMain.exe", "gray"),
    ("CUDA", "CUDAMain.exe", "gray"),
    ("OpenCL", "OpenCLMain.exe", "gray"),

    ("CPU", "CPUMain.exe", "blur"),
    ("CUDA", "CUDAMain.exe", "blur"),
    ("OpenCL", "OpenCLMain.exe", "blur"),

    ("CUDA optimized", "CUDAMain.exe", "blurO"),
    ("OpenCL optimized", "OpenCLMain.exe", "blurO"),
]


def run_program(executable, operation):
    result = subprocess.run(
        [str(ROOT / executable), operation],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{executable} {operation} failed:\n"
            f"{result.stdout}\n{result.stderr}"
        )

    if executable == "CPUMain.exe":
        pattern = r"CPU processing time:\s*([\d.]+)\s*ms"
    elif executable == "CUDAMain.exe":
        pattern = r"Kernel execution\s*:\s*([\d.]+)\s*ms"
    elif executable == "OpenCLMain.exe":
        pattern = r"Kernel time:\s*([\d.]+)\s*ms"
    else:
        raise ValueError(f"Unknown executable: {executable}")

    match = re.search(pattern, result.stdout)

    if not match:
        raise RuntimeError(
            f"Processing time not found in output of {executable}:\n"
            f"{result.stdout}\n{result.stderr}"
        )

    return float(match.group(1)), result.stdout


def main():
    results = []

    for name, executable, operation in TESTS:
        print(f"\nTesting {name}: {operation}")

        for _ in range(WARMUP_RUNS):
            run_program(executable, operation)

        times = []

        for i in range(RUNS):
            elapsed_ms, output = run_program(executable, operation)
            times.append(elapsed_ms)
            print(f"  Run {i + 1:02}: {elapsed_ms:.3f} ms")

        average = statistics.mean(times)
        median = statistics.median(times)
        deviation = statistics.stdev(times)

        print(f"  Average: {average:.3f} ms")
        print(f"  Median:  {median:.3f} ms")
        print(f"  Std dev: {deviation:.3f} ms")

        results.append({
            "implementation": name,
            "operation": operation,
            "runs": RUNS,
            "average_ms": average,
            "median_ms": median,
            "std_dev_ms": deviation,
        })

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "benchmark.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "implementation",
                "operation",
                "runs",
                "average_ms",
                "median_ms",
                "std_dev_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()