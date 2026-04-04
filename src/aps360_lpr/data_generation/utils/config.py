import yaml

class Configs():
    def __init__(self, file_dir):
        with open(file_dir / 'config.yaml', 'r') as file:
            config_content = yaml.safe_load(file)
        self.num_of_plates = config_content['data_generation']['num_of_plates']
        self.num_of_val = config_content['data_generation']['num_of_val']