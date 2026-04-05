"""
Shared utility functions for the RAG system.
Provides timing, logging, and output formatting helpers.
"""

import os
import time
import csv
import json
import functools
from datetime import datetime

from config import RESULTS_DIR, CACHE_DIR


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def timer(func):
    """Decorator that measures function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  [{func.__name__}] completed in {elapsed:.2f}s")
        return result, elapsed
    return wrapper


def measure_time(func, *args, **kwargs):
    """Measure execution time of a function call. Returns (result, seconds)."""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


def save_results_csv(results, filename="evaluation_results.csv"):
    """Save a list of result dicts to CSV."""
    ensure_dirs()
    filepath = os.path.join(RESULTS_DIR, filename)
    if not results:
        print("No results to save.")
        return filepath
    fieldnames = results[0].keys()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {filepath}")
    return filepath


def save_json(data, filename):
    """Save data as JSON to the results directory."""
    ensure_dirs()
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {filepath}")
    return filepath


def log(message, level="INFO"):
    """Simple timestamped logger."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def print_separator(title=""):
    """Print a visual separator for console output."""
    width = 70
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{'=' * padding} {title} {'=' * padding}")
    else:
        print(f"\n{'=' * width}")


def truncate_text(text, max_length=200):
    """Truncate text with ellipsis for display purposes."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
