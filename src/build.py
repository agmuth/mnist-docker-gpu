
from __future__ import print_function
import argparse
import torch

import torch.nn.functional as F

import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from src.cnn import CNN, img_transform
from utils import *
from src.constants import CONFIG_DIR, MODEL_DIR
import yaml
from schemas import ModelBuildConfig


def main(config_file: str = "mnist_cnn.yaml"): # run as cli
    with open(CONFIG_DIR.joinpath(config_file), 'r') as file:   
        args = yaml.safe_load(file)
        args = ModelBuildConfig(**args)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    trainer = Trainer(train_kwargs)
    tester = Tester(test_kwargs)

    model = CNN().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        trainer.train(args, model, device, optimizer, epoch)
        tester.test(model, device)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), MODEL_DIR.joinpath(args.file_name))


if __name__ == '__main__':
    main()