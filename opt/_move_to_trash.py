#!/usr/bin/env python3
"""
Helper script to move unused files to trash folder
"""
import os
import shutil

# Files to move
files_to_move = [
    'run_parallel_distzo2_dp_aggzo.py',
    'run_dpaggzo.py',
    'test_parallel_distzo2_dp_aggzo.py',
    'examples/parallel_distzo2_dp_aggzo.sh',
    'examples/dpaggzo.sh',
    'src/parallel_distzo2_dp_aggzo_wrapper_seq2seq.py',
    'src/ht_opt.py',
]

# Source and destination
source_dir = '/root/autodl-tmp/dpscal/opt'
dest_dir = '/root/autodl-tmp/trash/dpscal_opt_archived'

# Create destination directory
os.makedirs(dest_dir, exist_ok=True)

# Move files
for file_path in files_to_move:
    source = os.path.join(source_dir, file_path)
    dest = os.path.join(dest_dir, os.path.basename(file_path))
    
    if os.path.exists(source):
        print(f"Moving {file_path} -> {dest}")
        shutil.move(source, dest)
    else:
        print(f"Warning: {file_path} does not exist, skipping")

print("\n✓ Files moved successfully!")

