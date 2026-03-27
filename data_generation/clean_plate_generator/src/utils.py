import os
import shutil
import random
import string
from pathlib import Path


def _safe_copy_single_file(src, dst, *, follow_symlinks=True):
    """
    Internal helper function for single files: safely copies,
    skipping if the destination file already exists.
    """
    try:
        # Use 'xb' mode to ensure the file is created only if it doesn't exist
        with open(src, 'rb') as fsrc, open(dst, 'xb') as fdst:
            shutil.copyfileobj(fsrc, fdst)

        # Copy metadata (permissions, timestamps)
        shutil.copystat(src, dst, follow_symlinks=follow_symlinks)
        print(f'Copied: {dst}')

    except FileExistsError:
        print(f'Skipped (file already exists): {dst}')
    except Exception as e:
        print(f'Error copying [{dst}]: {e}')


def safe_copy(src_path, dst_path):
    """
    Smart safe copy function that preserves the source directory itself.
    """
    if not os.path.exists(src_path):
        print(f"Error: Source path '{src_path}' does not exist.")
        return

    # Get the base name of the source (e.g., 'my_folder' from '/path/to/my_folder')
    src_name = os.path.basename(os.path.abspath(src_path))

    # If the destination is an existing directory, we place the source INSIDE it.
    # Otherwise, we assume dst_path is the exact intended path.
    if os.path.exists(dst_path) and os.path.isdir(dst_path):
        actual_dst = Path(dst_path) / src_name
    else:
        actual_dst = dst_path

    if os.path.isdir(src_path):
        print(f'Copying directory: {src_path} -> {actual_dst}')

        # dirs_exist_ok=True allows merging if actual_dst already exists
        shutil.copytree(
            src_path,
            actual_dst,
            copy_function=_safe_copy_single_file,
            dirs_exist_ok=True,
        )
        print('Directory copy/merge completed.')

    else:
        # Handle single file copy
        dst_dir = os.path.dirname(actual_dst)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)

        _safe_copy_single_file(src_path, actual_dst)


def generate_plate(letter_num, number_num):
    letters = ''.join(random.choices(string.ascii_uppercase, k=letter_num))
    numbers = ''.join(random.choices(string.digits, k=number_num))
    return letters, numbers


def init_out(base_dir, out_path):
    Path(out_path).mkdir(parents=True, exist_ok=True)
    css_path = base_dir / 'src' / 'styles.css'
    font_path = base_dir / 'src' / 'fonts'
    assets_path = base_dir / 'src' / 'assets'
    safe_copy(css_path, out_path)
    safe_copy(font_path, out_path)
    safe_copy(assets_path, out_path)
