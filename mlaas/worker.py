import argparse
import os
import logging
import sys

import time
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from json_loader import JsonDataset

from models.svm import hinge_loss

from utils import init_logger, reset_logger_level, str_to_dict
from config import TRAINER_LOG_INTERVAL, PS_AVERAGE_EVERY_N

from server import get_ps

# logging.basicConfig(level=logging.DEBUG)
logger: logging.Logger = init_logger(
    'Worker', log_file='worker.log', log_stream=sys.stdout)

loss_func = None


class Worker(nn.Module):
    def __init__(self, rank, world_size, model, model_export_path, params):
        super(Worker, self).__init__()
        self.rank = rank
        self.ps_rref = rpc.remote('ps', get_ps, args=(
            self.rank, world_size, model, model_export_path, params))
        # print(type(self.ps_rref))
        assert self.ps_rref is not None, 'ps_rref is None'

    def get_param_rref(self):
        param_rref = self.ps_rref.rpc_sync().get_param_remote_ref(self.rank)
        return param_rref

    def forward(self, x):
        model_output = self.ps_rref.rpc_sync().model_forward(self.rank, x)
        return model_output

    def average_model(self):
        self.ps_rref.rpc_sync().average_models(self.rank)

    def record_loss(self, loss):
        self.ps_rref.rpc_sync().record_loss(self.rank, loss)

    def request_save_model(self):
        self.ps_rref.rpc_sync().save_model()

    def request_save_loss(self):
        self.ps_rref.rpc_sync().save_loss_log(self.rank)

    def get_current_workers(self):
        return self.ps_rref.rpc_sync().get_worker_num()

    def set_final_training(self):
        self.ps_rref.rpc_sync().set_final_training(self.rank)


def run_training_loop(rank, world_size, lr, iteration, train_loader, model, model_path, params):
    worker = Worker(rank, world_size, model, model_path, params)
    param_rref = worker.get_param_rref()
    opt = DistributedOptimizer(optim.SGD, param_rref, lr=lr)

    logger.info('Worker {rank} starting training')

    for i in range(iteration):
        logger.info(f'Worker {rank} at iteration {i}')
        for step, (data, target) in enumerate(train_loader):
            with dist_autograd.context() as cid:
                output = worker(data)
                loss = loss_func(output, target)
                dist_autograd.backward(cid, [loss])
                opt.step(cid)
            if step % PS_AVERAGE_EVERY_N == 0:
                logger.info(
                    f'Worker {rank} at iteration {i}, step {step}, average model')
                worker.average_model()
            if step % TRAINER_LOG_INTERVAL == 0:
                logger.info(f'Worker {rank} at iteration {i}, step {step}')
                logger.info(f'Worker {rank} loss: {loss.item()}')
                worker.record_loss(loss.item())

        logger.info(f'Worker {rank} at iteration {i}')
    if iteration > 1000:
        EXTRA_EPOCHS = 10
    else:
        EXTRA_EPOCHS = 1
    if rank == 1:
        # do extra work
        while worker.get_current_workers() != 1:
            time.sleep(1)
            continue
        worker.set_final_training()
        for i in range(EXTRA_EPOCHS):
            for step, (data, target) in enumerate(train_loader):
                with dist_autograd.context() as cid:
                    output = worker(data)
                    loss = loss_func(output, target)
                    dist_autograd.backward(cid, [loss])
                    opt.step(cid)
                if step % TRAINER_LOG_INTERVAL == 0:
                    logger.info(f'Worker {rank} at iteration {i}, step {step}')
                    logger.info(f'Worker {rank} loss: {loss.item()}')
                    worker.record_loss(loss.item())

            logger.info(f'Worker {rank} at extra iteration {i}')

    logger.info(f'Worker {rank} finished training')

    # Save the model
    worker.request_save_model()
    worker.request_save_loss()


def run_worker(rank, world_size, lr, iteration, train_loader, model, model_path, params):
    logger.info(f"Worker {rank} initializing")
    rpc.init_rpc(name=f'worker_{rank}', rank=rank, world_size=world_size)

    logger.info(f'Worker {rank} done initializing')
    logger.info(f'Worker {rank} starting training loop')
    run_training_loop(rank, world_size, lr, iteration,
                      train_loader, model, model_path, params)
    logger.info(f'Worker {rank} finished training loop')
    logger.info(f'Worker {rank} is shutting down')
    rpc.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parameter Server Worker Node')

    parser.add_argument('--rank', type=int, required=True,
                        help='Rank of the worker')
    parser.add_argument('--world_size', type=int,
                        required=True, help='Number of workers and ps')
    parser.add_argument('--master_addr', type=str,
                        required=True, help='Address of the master node')
    parser.add_argument('--master_port', type=str,
                        required=True, help='Port of the master node')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--model_export_name', type=str,
                        required=True, help='Model export name')
    parser.add_argument('--iterations', type=int,
                        default=1000, help='Iterations')
    parser.add_argument('--data', type=str, required=True, help='Data file')
    parser.add_argument('--datatype', type=str,
                        required=True, help='Data type')
    parser.add_argument('--params', type=str_to_dict,
                        required=True, help='Model Params')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    if args.debug:
        reset_logger_level(logger, logging.DEBUG)

    if str(args.model).upper() == 'SVM':
        logger.info('Using SVM model')
        loss_func = hinge_loss
    else:
        loss_func = F.cross_entropy

    os.environ['MASTER_ADDR'] = args.master_addr
    # os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    rank = args.rank

    if args.datatype == 'json':
        train_loader = DataLoader(JsonDataset(
            args.data), batch_size=args.batch_size, shuffle=True)
    elif args.datatype == 'MNIST':
        train_loader = DataLoader(
            datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])), batch_size=args.batch_size, shuffle=True
        )
    elif args.datatype == 'CIAR10':
        train_loader = DataLoader(
            datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])), batch_size=args.batch_size, shuffle=True
        )
    else:
        logger.critical('Unsupport data type!')

    # For testing purpose
    # p = mp.Process(target=run_worker, args=(rank, args.world_size, args.lr, train_loader, test_loader))
    # p.start()
    # p.join()
    run_worker(rank, args.world_size, args.lr,
               args.iterations, train_loader, args.model, args.model_export_name, args.params)
