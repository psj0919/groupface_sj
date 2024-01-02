from core.engine_groupface import Trainer
from Config.config import get_config_dict



if __name__ == '__main__':
    config = get_config_dict()
    tariner = Trainer(config)
    tariner.training()