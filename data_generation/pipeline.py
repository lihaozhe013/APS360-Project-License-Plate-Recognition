from pathlib import Path
from utils.split_and_copy import split_and_move
from utils.directory_manager import DirectoryManager
from utils.resize import resize_data
import shutil

base_dir = Path(__file__).parent.resolve()

clean_plate_generator_dir = base_dir / "clean_plate_generator"
domain_randomizer_dir = base_dir / "domain_randomizer"

clean_plate_out = base_dir / "clean_plate_generator" / "out"
domain_random_input_dir = base_dir / "domain_randomizer" / "clean_plates"
domain_random_output_dir = base_dir / "domain_randomizer" / "aged_plates"
train_data_dir = base_dir / ".." / "train" / "data"
train_set_dir = train_data_dir / "train"
val_set_dir = train_data_dir / "val"

clean_list = [
    clean_plate_out,
    domain_random_output_dir,
    domain_random_input_dir,
    train_data_dir,
]


def main():
    builder = DirectoryManager()

    # clean
    builder.clean(clean_list)

    # generate clean plates
    builder.run(clean_plate_generator_dir, "uv run generate.py")
    builder.retain_only_extensions(clean_plate_out, ".jpg")
    builder.delete(clean_plate_out / "assets")
    builder.delete(clean_plate_out / "fonts")

    # domain_randomize
    builder.move(clean_plate_out, domain_random_input_dir)
    builder.run(domain_randomizer_dir, "uv run process_plates.py")

    # Split and distribute to train/val folders
    split_and_move(domain_random_output_dir, train_set_dir, val_set_dir, val_count=20)

    # copy real val data into val folder
    shutil.copytree(base_dir / "real_val_data", val_set_dir, dirs_exist_ok=True)

    resize_data(train_data_dir)

    print("\Script Finished Successfully!")


if __name__ == "__main__":
    main()
