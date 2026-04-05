from pathlib import Path
from aps360_lpr.data_generation.utils.split_and_copy import split_and_move
from dm.directory_manager import DirectoryManager
from aps360_lpr.data_generation.utils.resize import resize_data
from aps360_lpr.data_generation.utils.config import Configs
import shutil


scripts_base_dir = Path(__file__).parent.resolve()
base_dir = scripts_base_dir / '..' / '..' / '..'
dataset_base_dir = base_dir / 'dataset'
config = Configs(scripts_base_dir)

clean_plate_generator_dir = scripts_base_dir / 'clean_plate_generator'
background_embedder_dir = scripts_base_dir / 'background_embedder'
domain_randomizer_dir = scripts_base_dir / 'domain_randomizer'

clean_plate_out = dataset_base_dir / 'clean_plate_out'
background_embedder_output_dir = dataset_base_dir / 'background_embedder_out_'
domain_random_output_dir = dataset_base_dir / 'temp' / 'domain_randomizer_out_'
train_set_dir = dataset_base_dir / 'train'
val_set_dir = dataset_base_dir / 'val'

clean_list = [dataset_base_dir]

def main():
    dm = DirectoryManager()

    # clean
    dm.clean(clean_list)

    # generate clean plates
    dm.run(clean_plate_generator_dir, ['uv', 'run', 'generate.py'])
    dm.retain_only_extensions(clean_plate_out, '.jpg')
    dm.delete(clean_plate_out / 'assets')
    dm.delete(clean_plate_out / 'fonts')

    # domain_randomize
    dm.run(domain_randomizer_dir, ['uv', 'run', 'process_plates.py'])

    # background_embedder
    dm.run(background_embedder_dir, ['uv', 'run', 'background_embedder.py'])

    dm.delete(domain_random_output_dir)
    print(f'Data Generation Pipeline Finished Successfully!')


if __name__ == '__main__':
    main()
