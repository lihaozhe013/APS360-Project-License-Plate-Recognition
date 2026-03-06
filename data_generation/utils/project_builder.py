import os
import shutil
import subprocess
from pathlib import Path

class ProjectBuilder:
    def clean_dirs(self, base_dir, directories):
        """
        Step 1: Clean specific directories. 
        If they don't exist, simply skip them without error.
        """
        print("\n>>> Task: Cleaning Directories")
        for d in directories:
            target = base_dir / d
            if target.exists() and target.is_dir():
                # We use shutil.rmtree to remove the folder and all its contents
                shutil.rmtree(target)
                print(f"Removed: {d}")
            else:
                print(f"Skipped (not found): {d}")

    def run_command(self, work_dir, command):
        """
        Step 2: Execute build commands.
        check=True ensures a CalledProcessError is raised if the command fails.
        """
        print(f"\n>>> Task: Running Build Command -> '{command}' in {work_dir}")
        cwd_path = work_dir
        
        # subprocess.run will raise an exception if the return code is non-zero
        subprocess.run(command, shell=True, cwd=cwd_path, check=True)

    def copy_and_rename(self, base_dir, src, dst):
        """
        Step 3: Copy/Rename files or folders.
        No internal try-catch here; if it fails, it bubbles up to main().
        """
        src_path = base_dir / src
        dst_path = base_dir / dst

        print(f"\n>>> Task: Copying/Renaming -> {src} to {dst}")

        # Ensure the destination's parent directory exists (auto-create)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if src_path.is_dir():
            # copytree copies the directory and allows 'dst' to be the new name
            shutil.copytree(src_path, dst_path)
        else:
            # copy2 preserves metadata for files
            shutil.copy2(src_path, dst_path)

    def retain_only_extensions(self, directory, extension):
        """
        Delete all files in a directory except those with the specified extension.
        """
        target = Path(directory)
        print(f"\n>>> Task: Retaining only {extension} files in {target}")

        if not target.exists() or not target.is_dir():
            print(f"Skipped (not found or not a dir): {target}")
            return

        removed_count = 0
        for item in target.iterdir():
            if item.is_file():
                if item.suffix.lower() != extension.lower():
                    try:
                        item.unlink()
                        removed_count += 1
                    except Exception as e:
                        print(f"Failed to delete {item}: {e}")

        print(f"Removed {removed_count} files.")

    def smart_delete(self, path):
        """
        Smart delete: removes a file or directory recursively given a path.
        """
        target = Path(path)
        print(f"\n>>> Task: Smart Deleting -> {target}")

        if not target.exists():
            print(f"Skipped (not found): {target}")
            return

        try:
            if target.is_dir():
                shutil.rmtree(target)
                print(f"Removed directory: {target}")
            elif target.is_file():
                target.unlink()
                print(f"Removed file: {target}")
        except Exception as e:
            print(f"Failed to delete {target}: {e}")