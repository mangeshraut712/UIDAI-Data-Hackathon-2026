#!/usr/bin/env python3
"""
UIDAI Aadhaar Analysis Pipeline - Premium Executive Edition
Orchestrates complete analysis workflow for the 2026 Hackathon
"""

import sys
import time
from pathlib import Path
import subprocess
import warnings

# Silence non-critical library warnings in the main orchestrator
warnings.filterwarnings('ignore')

# ANSI Color Codes for Premium Terminal Output
C = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "CYAN": "\033[96m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
    "END": "\033[0m",
}

def check_environment():
    """Verify Python version and dependencies"""
    print(f"{C['CYAN']}Checking environment...{C['END']}")
    
    python_version = sys.version_info
    if python_version < (3, 9):
        print(f"{C['RED']}Error: Python {python_version.major}.{python_version.minor} detected{C['END']}")
        print("Requires Python 3.9 or higher")
        return False
    
    print(f"{C['GREEN']}Python {python_version.major}.{python_version.minor}.{python_version.micro} - [OK]{C['END']}")
    return True


def execute_module(module_path, description):
    """Run analysis module and capture output"""
    print(f"\n{C['BOLD']}{C['BLUE']}{'='*80}{C['END']}")
    print(f"{C['BOLD']}RUNNING: {description}{C['END']}")
    print(f"{C['BLUE']}{'='*80}{C['END']}")
    
    start = time.time()
    
    try:
        # Run with real-time output stream for better user experience
        process = subprocess.Popen(
            [sys.executable, str(module_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output line by line as it happens
        for line in process.stdout:
            print(f"  {line.strip()}")
            
        process.wait(timeout=600)  # Increased timeout for NNs
        
        duration = time.time() - start
        if process.returncode == 0:
            print(f"\n{C['GREEN']}[DONE] SUCCESS:{C['END']} {description}")
            print(f"{C['CYAN']}Duration:{C['END']} {duration:.1f}s")
            return True
        else:
            print(f"\n{C['RED']}[FAIL] FAILED:{C['END']} {description}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n{C['RED']}[TIME] TIMEOUT:{C['END']} {description} exceeded 10 minutes")
        return False
    except Exception as e:
        print(f"\n{C['RED']}[ERR] ERROR:{C['END']} {str(e)}")
        return False


def main():
    """Execute complete analysis pipeline"""
    print("\n" + f"{C['HEADER']}{C['BOLD']}{'='*80}{C['END']}")
    print(f"{C['HEADER']}{C['BOLD']}         UIDAI AADHAAR ANALYSIS PIPELINE - 2026 EDITION{C['END']}")
    print(f"{C['HEADER']}{C['BOLD']}{'='*80}{C['END']}\n")
    
    if not check_environment():
        sys.exit(1)
    
    base_dir = Path(__file__).parent
    
    # Analysis pipeline configuration
    pipeline = [
        (
            base_dir / "src/analysis/comprehensive_analysis.py",
            "Initial Data Loading & Statistical Rigor"
        ),
        (
            base_dir / "src/analysis/advanced_analytics.py",
            "Advanced Machine Learning Suite"
        ),
        (
            base_dir / "src/analysis/iit_level_analytics.py",
            "IIT-Level Advanced Modeling (t-SNE/Prophet/Monte-Carlo)"
        ),
        (
            base_dir / "src/visualization/premium_visualizations.py",
            "Publication-Quality Chart Generation (01-08)"
        ),
        (
            base_dir / "src/visualization/wow_factor.py",
            "Interactive Dashboards & Policy ROI"
        ),
        (
            base_dir / "src/visualization/generate_infographic.py",
            "Executive Summary Infographic"
        ),
        (
            base_dir / "src/visualization/interactive_dashboard.py",
            "Deployment-Ready HTML Dashboard"
        ),
    ]
    
    # Execute pipeline
    results = []
    total_start = time.time()
    
    for module_path, description in pipeline:
        if not module_path.exists():
            print(f"{C['YELLOW']}! Warning: {module_path.name} not found, skipping{C['END']}")
            continue
        
        success = execute_module(module_path, description)
        results.append((description, success))
    
    # Summary
    print("\n" + f"{C['BOLD']}{'='*80}{C['END']}")
    print(f"{C['BOLD']}                         FINAL PIPELINE SUMMARY{C['END']}")
    print(f"{C['BOLD']}{'='*80}{C['END']}")
    
    for description, success in results:
        status = f"{C['GREEN']}SUCCESS{C['END']}" if success else f"{C['RED']}FAILED {C['END']}"
        print(f"[{status}] | {description}")
    
    total_duration = time.time() - total_start
    success_count = sum(1 for _, success in results if success)
    
    print(f"\n{C['BOLD']}Final Status:{C['END']} {success_count}/{len(results)} Modules Succeeded")
    print(f"{C['BOLD']}Total Execution Time:{C['END']} {total_duration/60:.1f} minutes")
    
    if success_count == len(results):
        print(f"\n{C['BOLD']}{C['GREEN']}COMPLETED: ALL SYSTEMS GO! YOUR SUBMISSION IS READY FOR THE 1ST PRIZE!{C['END']}\n")
        return 0
    else:
        print(f"\n{C['BOLD']}{C['YELLOW']}! SOME MODULES REQUIRING ATTENTION OR TIMED OUT.{C['END']}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
