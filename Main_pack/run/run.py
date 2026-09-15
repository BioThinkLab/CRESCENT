#!/usr/bin/env python3
"""CRESCENT user configuration and entry point."""

from pipeline import run_pipeline


# Folder names under Main_pack/Data/input.
PROJECT_NAMES = ["EXAMPLE"]

# "amp" for amplification or "del" for deletion.
MUTATION_TYPE = "del"

# Leave empty to analyze every available chromosome.
# Examples: [1, 2, 8], ["1", "X"], or ["chr1", "chrX"].
CHROMOSOME_NUMBERS = []

# Set True to remove arm-level events before analysis.
ONLY_USE_FOCAL = False

# Probability threshold used to produce the final segments.
CLASSIFICATION_THRESHOLD = 0.5

# Process samples in small batches to limit temporary disk use.
USE_CHUNKING = True

# Used only when chunking is enabled.
SAMPLES_PER_CHUNK = 500

# Warn, but continue, when a relevant drive has less free space than this.
LOW_DISK_WARNING_GB = 15


if __name__ == "__main__":
    run_pipeline(
        project_names=PROJECT_NAMES,
        mutation_type=MUTATION_TYPE,
        chromosome_numbers=CHROMOSOME_NUMBERS,
        only_use_focal=ONLY_USE_FOCAL,
        classification_threshold=CLASSIFICATION_THRESHOLD,
        use_chunking=USE_CHUNKING,
        samples_per_chunk=SAMPLES_PER_CHUNK,
        low_disk_warning_gb=LOW_DISK_WARNING_GB,
    )
