import os
import sys
from pathlib import Path
from utils.project_builder import ProjectBuilder

base_dir = Path(__file__).parent.resolve()

clean_plate_generator_dir = base_dir / 'clean_plate_generator'
domain_randomizer_dir = base_dir / 'domain_randomizer'

clean_plate_out = base_dir / 'clean_plate_generator' / 'out'
domain_random_input_dir = base_dir / 'domain_randomizer' / 'clean_plates'
domain_random_output_dir = base_dir / 'domain_randomizer' / 'aged_plates'
train_data_dir = base_dir / '..' / 'train' / 'data'

clean_list = [
    clean_plate_out, 
    domain_random_output_dir, 
    domain_random_input_dir
]

def main():
    builder = ProjectBuilder()

    try:
        # clean
        builder.clean_dirs(base_dir, clean_list)

        # generate clean plates
        builder.run_command(clean_plate_generator_dir, "python generate.py")
        builder.retain_only_extensions(clean_plate_out, '.jpg')
        builder.smart_delete(clean_plate_out / 'assets')
        builder.smart_delete(clean_plate_out / 'fonts')

        # domain_randomize
        builder.copy_and_rename(clean_plate_out, domain_random_input_dir)
        builder.run_command(domain_randomizer_dir, "python process_plates.py")
        builder.copy_and_rename(domain_random_output_dir, train_data_dir)

        print("\nBuild Script Finished Successfully!")

    except Exception as error:
        # Any failure in run_command or copy_and_rename lands here
        print("\nCRITICAL ERROR: Script terminated.")
        print(f"Details: {error}")
        sys.exit(1)

if __name__ == "__main__":
    main()