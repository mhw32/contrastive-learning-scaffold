import os
import torch
from copy import deepcopy
from src.agents.agents import *
from src.utils.setup import process_config
from src.utils.utils import load_json


def run(config_path, gpu_device=-1):
    config = process_config(config_path)
    if gpu_device >= 0:
        config.gpu_device = [gpu_device]
    AgentClass = globals()[config.agent]
    agent = AgentClass(config)

    if config.continue_exp_dir is not None:
        agent.logger.info("Found existing model... Continuing training!")
        checkpoint_dir = os.path.join(config.continue_exp_dir, 'checkpoints')
        agent.load_checkpoint(
            config.continue_exp_name,
            checkpoint_dir=checkpoint_dir, 
            load_memory_bank=True, 
            load_model=True,
            load_optim=True,
            load_epoch=True,
        )

    try:
        agent.run()
        agent.finalise()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='path to config file')
    parser.add_argument('--gpu-device', type=int, default=-1)
    args = parser.parse_args()

    run(args.config, args.gpu_device)
