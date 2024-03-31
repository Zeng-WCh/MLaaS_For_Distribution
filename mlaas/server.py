import argparse
import json
import os
from threading import Lock
import logging
import sys

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from typing import Callable, Dict

from model_config import get_model_creator
from utils import init_logger, reset_logger_level, str_to_dict

create_model_fn: Callable | None = None
params: Dict | None = None

# logging.basicConfig(level=logging.DEBUG)
logger: logging.Logger = init_logger(
    'Parameter Server', log_file='ps.log', log_stream=sys.stdout)


class ParameterServer(object):
    def __init__(self, world_size, model, model_saved_path: str, parms):
        self.workers = dict()
        self.lock = Lock()
        self.device = torch.device('cpu')
        self.create_model_fn = get_model_creator(model)
        self.worker_lossess = dict()
        self.args = list()
        self.kwargs = parms
        self.world_size = world_size
        self.path = model_saved_path
        self.loss_file_base = model_saved_path.replace(
            os.path.basename(model_saved_path), '')
        self.final_train = False
        self.final_train_idx = None

    def model_forward(self, workder_rank, x):
        logger.info(f'Worker {workder_rank} is doing forward...')
        return self.workers[workder_rank](x)

    def get_distributed_gradients(self, context_id):
        grads = dist_autograd.get_gradients(context_id)

        grad = dict()

        for k, v in grads.items():
            k_cpu = k.to('cpu')
            v_cpu = v.to('cpu')
            grad[k_cpu] = v_cpu

        return grad

    def get_param_remote_ref(self, worker_rank):
        return [rpc.RRef(param) for param in self.workers[worker_rank].parameters()]

    def create_model(self, worker_rank):
        with self.lock:
            if worker_rank not in self.workers.keys():
                logger.info(f'Worker {worker_rank} is creating a model...')
                self.workers[worker_rank] = self.create_model_fn(
                    self.args, self.kwargs)
                self.worker_lossess[worker_rank] = list()

    '''
    average the model for worker_rank
    '''

    def average_models(self, worker_rank):
        logger.info(f'Worker {worker_rank} is averaging the model...')
        state_dict = self.workers[worker_rank].state_dict()

        with self.lock:
            for k in state_dict:
                state_dict[k] = sum(self.workers[r].state_dict()[k]
                                    for r in self.workers.keys()) / len(self.workers)

        self.workers[worker_rank].load_state_dict(state_dict)

    def record_loss(self, worker_rank, loss):
        logger.info(f'Worker {worker_rank} is logging loss{loss}...')
        self.worker_lossess[worker_rank].append(loss)

    def save_model(self):
        self.world_size -= 1
        if self.world_size == 1:
            logger.info('Saving model...')
            if not self.final_train:
                # Average all the models
                k = list(self.workers.keys())[0]
                self.average_models(k)
                with self.lock:
                    torch.save(self.workers[k].state_dict(), self.path)

            else:
                with self.lock:
                    torch.save(
                        self.workers[self.final_train_idx].state_dict(), self.path)
            logger.info(f'Model saved to {self.path}!')

    def save_loss_log(self, worker_rank):
        filename = os.path.join(self.loss_file_base,
                                f'worker_{worker_rank}_loss.log')

        with open(filename, 'w') as f:
            for loss in self.worker_lossess[worker_rank]:
                f.write(f'{loss}\n')

    def get_worker_num(self):
        return self.world_size - 1

    def set_final_training(self, rank):
        self.average_models(rank)
        self.final_train = True
        self.final_train_idx = rank


ps: ParameterServer = None
ps_lock = Lock()


def get_ps(rank, world_size, model, model_export_path, params):
    global ps
    with ps_lock:
        if ps is None:
            logger.info("Creating PS...")
            ps = ParameterServer(world_size, model, model_export_path, params)
            logger.info("PS created!")
        logger.info(f'Registering model for worker {rank}...')
        ps.create_model(rank)
        return ps


def start_ps(rank, world_size, timeout: float = 0):
    if create_model_fn is None:
        logger.critical('Model creator is not provided, exiting...')
        return
    logger.info("PS Node init RPC")
    if timeout != 0:
        # rpc._set_rpc_timeout(timeout)
        logger.info(f"Set RPC timeout to {timeout}s")
    # backend = rpc.BackendType.TENSORPIPE
    # backend_options = rpc.TensorPipeRpcBackendOptions(rpc_timeout=timeout)
    rpc.init_rpc(
        name='ps',
        rank=rank,
        world_size=world_size,
        # rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
        #     rpc_timeout=timeout),
        # backend=backend,
        # rpc_backend_options=backend_options
    )
    logger.info("RPC successfully initialized! PS is running...")
    rpc.shutdown()
    logger.info("PS is shutdown!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parameter Server Node')

    parser.add_argument('--world_size', type=int, default=4,
                        help='Number of workers and ps')
    parser.add_argument('--port', type=int, default=29500,
                        help='RPC Port number')
    parser.add_argument('--timeout', type=float, default=0,
                        help='RPC Timeout, 0 means no timeout')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--model_export_name', type=str,
                        required=True, help='Model export path')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--params', type=str_to_dict, help='Model parameters')

    args = parser.parse_args()

    if args.debug:
        reset_logger_level(logging.DEBUG)

    create_model_fn = get_model_creator(args.model)
    params = args.params

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)

    # mp.set_start_method("spawn")
    # p = mp.Process(target=start_ps, args=(0, args.world_size))
    # p.start()
    # p.join()
    # ps = ParameterServer(create_model_fn, args.params)
    start_ps(0, args.world_size, args.timeout)

    # logger.info('Before shutdown, saving model...')
    # path = {args.model_export_name}
    # ps.save_model(path)
    # logger.info(f'Model saved!')
    # logger.debug('Model saved to {path}!')
