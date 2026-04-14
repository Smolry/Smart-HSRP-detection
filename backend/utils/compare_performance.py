"""
PERFORMANCE COMPARISON TOOL
===========================
Compare original vs optimized pipeline performance

Usage:
    python compare_performance.py --video test.mp4 --frames 100
"""

import time
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Import both versions for comparison
from backend.core.video_pipeline import process_video as process_video_optimized


def run_benchmark(video_path, max_frames=100, frame_skip_values=[1, 2, 3]):
    """
    Benchmark different configurations
    """
    
    results = {
        "video": str(video_path),
        "max_frames": max_frames,
        "tests": []
    }
    
    print("=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Frames to process: {max_frames}")
    print()
    
    for skip in frame_skip_values:
        print(f"\n--- Testing with frame_skip={skip} ---")
        
        start_time = time.time()
        
        try:
            output = process_video_optimized(
                video_path=str(video_path),
                max_frames=max_frames,
                frame_skip=skip,
                enable_tracking=True,
                enable_ocr_stabilization=True
            )
            
            elapsed = time.time() - start_time
            
            # Extract metrics
            metadata = output.get("metadata", {})
            frames_processed = metadata.get("total_frames_processed", 0)
            avg_fps = metadata.get("avg_fps", 0)
            avg_frame_time = metadata.get("avg_frame_time_ms", 0)
            
            # Count detections
            total_vehicles = 0
            total_plates = 0
            total_violations = 0
            
            for frame in output.get("frames", []):
                entities = frame.get("entities", {})
                enforcements = frame.get("enforcements", {})
                
                total_vehicles += len(entities.get("vehicles", []))
                total_plates += len(entities.get("plates", []))
                total_violations += sum(
                    1 for tw in enforcements.get("two_wheelers", [])
                    if tw.get("helmet", {}).get("final_status") == "NO_HELMET"
                )
            
            test_result = {
                "frame_skip": skip,
                "elapsed_seconds": round(elapsed, 2),
                "frames_processed": frames_processed,
                "avg_fps": round(avg_fps, 2),
                "avg_frame_time_ms": round(avg_frame_time, 2),
                "total_vehicles": total_vehicles,
                "total_plates": total_plates,
                "total_violations": total_violations
            }
            
            results["tests"].append(test_result)
            
            # Print results
            print(f"✓ Completed in {elapsed:.2f}s")
            print(f"  Frames processed: {frames_processed}")
            print(f"  Average FPS: {avg_fps:.2f}")
            print(f"  Avg frame time: {avg_frame_time:.2f}ms")
            print(f"  Detections: {total_vehicles} vehicles, {total_plates} plates")
            print(f"  Violations: {total_violations}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            results["tests"].append({
                "frame_skip": skip,
                "error": str(e)
            })
    
    return results


def plot_results(results, output_dir):
    """
    Generate performance comparison plots
    """
    
    tests = [t for t in results["tests"] if "error" not in t]
    
    if not tests:
        print("No successful tests to plot")
        return
    
    frame_skips = [t["frame_skip"] for t in tests]
    fps_values = [t["avg_fps"] for t in tests]
    frame_times = [t["avg_frame_time_ms"] for t in tests]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Smart HSRP Performance Benchmark', fontsize=16, fontweight='bold')
    
    # Plot 1: FPS comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(frame_skips)), fps_values, color=['#2196F3', '#4CAF50', '#FF9800'])
    ax1.set_xlabel('Frame Skip')
    ax1.set_ylabel('Frames Per Second')
    ax1.set_title('Processing Speed (FPS)')
    ax1.set_xticks(range(len(frame_skips)))
    ax1.set_xticklabels([f'skip={s}' for s in frame_skips])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, fps_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Frame time comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(frame_skips)), frame_times, color=['#2196F3', '#4CAF50', '#FF9800'])
    ax2.set_xlabel('Frame Skip')
    ax2.set_ylabel('Milliseconds')
    ax2.set_title('Average Frame Processing Time')
    ax2.set_xticks(range(len(frame_skips)))
    ax2.set_xticklabels([f'skip={s}' for s in frame_skips])
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars2, frame_times)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Speedup comparison
    ax3 = axes[1, 0]
    baseline_fps = fps_values[0] if fps_values else 1
    speedups = [fps / baseline_fps for fps in fps_values]
    bars3 = ax3.bar(range(len(frame_skips)), speedups, color=['#2196F3', '#4CAF50', '#FF9800'])
    ax3.set_xlabel('Frame Skip')
    ax3.set_ylabel('Speedup Factor')
    ax3.set_title(f'Speedup vs Baseline (skip={frame_skips[0]})')
    ax3.set_xticks(range(len(frame_skips)))
    ax3.set_xticklabels([f'skip={s}' for s in frame_skips])
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline')
    ax3.grid(axis='y', alpha=0.3)
    ax3.legend()
    
    for i, (bar, val) in enumerate(zip(bars3, speedups)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Detection summary
    ax4 = axes[1, 1]
    vehicles = [t["total_vehicles"] for t in tests]
    plates = [t["total_plates"] for t in tests]
    violations = [t["total_violations"] for t in tests]
    
    x = np.arange(len(frame_skips))
    width = 0.25
    
    ax4.bar(x - width, vehicles, width, label='Vehicles', color='#2196F3')
    ax4.bar(x, plates, width, label='Plates', color='#4CAF50')
    ax4.bar(x + width, violations, width, label='Violations', color='#F44336')
    
    ax4.set_xlabel('Frame Skip')
    ax4.set_ylabel('Count')
    ax4.set_title('Detection Summary')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'skip={s}' for s in frame_skips])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "performance_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_path}")
    
    plt.close()


def generate_report(results, output_dir):
    """
    Generate markdown report
    """
    
    report_path = output_dir / "benchmark_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Smart HSRP Performance Benchmark Report\n\n")
        f.write(f"**Video:** {results['video']}\n")
        f.write(f"**Frames Tested:** {results['max_frames']}\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Frame Skip | FPS | Frame Time (ms) | Speedup | Vehicles | Plates | Violations |\n")
        f.write("|------------|-----|-----------------|---------|----------|--------|------------|\n")
        
        baseline_fps = None
        
        for test in results["tests"]:
            if "error" in test:
                continue
            
            fps = test["avg_fps"]
            if baseline_fps is None:
                baseline_fps = fps
            
            speedup = fps / baseline_fps if baseline_fps else 1.0
            
            f.write(
                f"| {test['frame_skip']} | "
                f"{fps:.2f} | "
                f"{test['avg_frame_time_ms']:.2f} | "
                f"{speedup:.2f}x | "
                f"{test['total_vehicles']} | "
                f"{test['total_plates']} | "
                f"{test['total_violations']} |\n"
            )
        
        f.write("\n## Recommendations\n\n")
        
        # Find best configuration
        tests = [t for t in results["tests"] if "error" not in t]
        if tests:
            best_fps = max(tests, key=lambda t: t["avg_fps"])
            best_balanced = [t for t in tests if t["frame_skip"] == 2]
            
            f.write("### For Real-Time Processing\n")
            f.write(f"- **Configuration:** frame_skip={best_fps['frame_skip']}\n")
            f.write(f"- **Performance:** {best_fps['avg_fps']:.2f} FPS\n")
            f.write(f"- **Use when:** Speed is critical\n\n")
            
            if best_balanced:
                f.write("### For Balanced Processing\n")
                f.write(f"- **Configuration:** frame_skip=2\n")
                f.write(f"- **Performance:** {best_balanced[0]['avg_fps']:.2f} FPS\n")
                f.write(f"- **Use when:** Good balance of speed and accuracy needed\n\n")
            
            f.write("### For Maximum Accuracy\n")
            f.write(f"- **Configuration:** frame_skip=1\n")
            f.write(f"- **Performance:** {tests[0]['avg_fps']:.2f} FPS\n")
            f.write(f"- **Use when:** Enforcement or legal requirements\n\n")
    
    print(f"✓ Saved benchmark report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Smart HSRP performance')
    parser.add_argument('--video', type=str, required=True, help='Path to test video')
    parser.add_argument('--frames', type=int, default=100, help='Max frames to process')
    parser.add_argument('--output', type=str, default='benchmark_results', help='Output directory')
    parser.add_argument('--skip-values', nargs='+', type=int, default=[1, 2, 3],
                       help='Frame skip values to test')
    
    args = parser.parse_args()
    
    # Setup
    video_path = Path(args.video)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return
    
    # Run benchmark
    results = run_benchmark(
        video_path=video_path,
        max_frames=args.frames,
        frame_skip_values=args.skip_values
    )
    
    # Save raw results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved raw results: {results_path}")
    
    # Generate visualizations
    try:
        plot_results(results, output_dir)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    # Generate report
    generate_report(results, output_dir)
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
