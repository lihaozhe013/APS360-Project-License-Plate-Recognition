import subprocess
import sys
import os
from pathlib import Path

def run_script(script_path, *args, cwd=None):
    print(f"\n{'='*70}")
    print(f"Executing: {script_path.name}")
    print(f"{'='*70}")
    
    cmd = [sys.executable, str(script_path)] + list(args)
    result = subprocess.run(cmd, cwd=cwd)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Script {script_path.name} failed with exit code {result.returncode}. Stopping pipeline.")
        sys.exit(result.returncode)

def main():
    # Root directory of the repository
    project_root = Path(__file__).parent.resolve()
    
    # 1. Data Generation Script
    data_pipeline = project_root / "src" / "aps360_lpr" / "data_generation" / "pipeline.py"
    
    # 2. Bounding Box Training Script
    bbox_train = project_root / "src" / "aps360_lpr" / "train" / "src" / "bbox_model" / "train_bbox.py"
    
    # 3. Train Pipeline Script (Bbox Crop + CRNN Train)
    train_pipeline = project_root / "src" / "aps360_lpr" / "train" / "src" / "train_pipeline.py"
    
    # Check if files exist
    for script in [data_pipeline, bbox_train, train_pipeline]:
        if not script.exists():
            print(f"[ERROR] Could not find script: {script}")
            sys.exit(1)

    print("\nStarting End-To-End LPR Pipeline...")

    # Step 1: Run Data Generation Pipeline
    # CWD is kept as project root or the script's parent depending on how the scripts are written.
    # The pipeline script has logic to resolve from __file__, so CWD doesn't strictly matter, 
    # but we'll set it to the script's dir for safety.
    run_script(data_pipeline, cwd=str(data_pipeline.parent))

    # Step 2: Train the Bounding Box CNN Model
    # The train_bbox.py script saves weights to "weights/" and looks for dataset via relativity,
    # so we MUST set cwd to its parent folder.
    run_script(bbox_train, cwd=str(bbox_train.parent))

    # Step 3: Use trained BBox model to crop plates, and feed to CRNN for training
    # The train_pipeline.py script imports local Modules, so setting CWD is crucial.
    run_script(train_pipeline, cwd=str(train_pipeline.parent))

    print(f"\n{'='*70}")
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
