import argparse
import sys
import subprocess
import logging
import os
import json
import socket
import threading
import requests

from argparse import ArgumentParser
from flask import Flask, request
from utils import get_network_interface, init_logger, reset_logger_level, get_broadcast_address
from multiprocessing import Queue, Process
from config import BROADCAST_PORT

arguments: None | argparse.Namespace = None
app = Flask('worker_controller')
app.config['UPLOAD_FOLDER'] = './training_data'

# logging.basicConfig(level=logging.DEBUG)
logger: logging.Logger = init_logger(
    'Worker Controller', log_file='worker_controller.log', log_stream=sys.stdout)

working_queue = Queue()


def register_worker(payload: dict, url: str):
    reponse = requests.post(url, json=payload)
    if reponse.status_code != 200:
        logger.error(f'Error when registering worker: {reponse.text}')
    logger.info('Worker registered!')


def worker_connector(client_ip, netmask, client_port):
    # addr = get_broadcast_address(client_ip, netmask)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(('', BROADCAST_PORT))

    payload = {
        'ip': client_ip,
        'port': client_port
    }

    # payload = json.dumps(payload).encode()

    while True:
        data = sock.recv(1024)

        data = json.loads(data.decode())
        logger.debug(f'Receive data: {data}')
        if data['type'] == 'worker_discovery':
            server_ip = data['ip']
            server_port = data['port']
            endpoint = data['endpoint']

            url = f'http://{server_ip}:{server_port}/{endpoint}'
            logger.info('Trying to register worker to server...')

            logger.debug(
                f'Server IP: {server_ip}, Server Port: {server_port}, Endpoint: {endpoint}')
            # Create a new thread to send HTTP request to the server
            t = threading.Thread(target=register_worker, args=(payload, url))
            t.run()
    sock.close()


def consumer(msgq: Queue):
    global arguments
    env = os.environ.copy()
    env['GLOO_SOCKET_IFNAME'] = arguments.network_interface
    env['TP_SOCKET_IFNAME'] = arguments.network_interface
    while True:
        logger.debug(f'Waiting for data...')
        json_data = msgq.get()
        if json_data == 'exit':
            break
        logger.debug(f'Get data: {json_data}')

        params = json_data['params']

        p = subprocess.Popen(
            [
                'python3', 'worker.py',
                '--rank', str(json_data['rank']),
                '--world_size', str(json_data['world_size']),
                '--master_addr', str(json_data['server_ip']),
                '--master_port', str(json_data['server_rpc_port']),
                '--lr', str(params['lr']),
                '--batch_size', str(params['batch_size']),
                '--iterations', str(params['iterations']),
                '--data', f'{app.config["UPLOAD_FOLDER"]}/{json_data["filename"]}',
                '--datatype', str(json_data['filetype']),
                '--model', str(json_data['model']),
                '--model_export_name', str(json_data['model_export_name']),
                '--params', json.dumps(params),
            ],
            env=env
        )

        p.wait()


@app.route('/api/create_worker', methods=['POST'])
def create_worker():
    if request.method != 'POST':
        return 'Invalid method'

    json_data = request.get_json()

    logger.debug(f'Receive from ip: {request.remote_addr}')
    logger.debug(f'JSON Data: {json_data}')
    logger.debug(f'Put data into queue')
    working_queue.put(json_data)

    return 'OK'


@app.route('/api/upload', methods=['POST'])
def file_upload():
    if request.method != 'POST':
        return 'Invalid method'
    if 'file' not in request.files:
        logger.info(
            f'Request from {request.remote_addr} does not have file part')
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        logger.info(
            f'Request from {request.remote_addr} does not have file name')
        return 'No selected file'

    if file:
        filename = file.filename
        logger.info(f'Receiving file {filename} from {request.remote_addr}')
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        logger.info(f'File {filename} saved')
        return 'OK'


if __name__ == '__main__':
    parser = ArgumentParser('Worker Controller')

    parser.add_argument('--port', type=int, default=8080,
                        help='Controller API Port')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    # parser.add_argument('--server_ip', type=str, required=True, help='Server IP')
    # parser.add_argument('--server_rpc_port', type=int, required=True, help='Server RPC Port')
    parser.add_argument('--network_interface', type=str,
                        default=None, help='RPC Network Interface')

    args = parser.parse_args()
    net_ifs = get_network_interface()

    if args.network_interface is None or args.network_interface == '':
        logger.critical('Network Interface is not specified')
        logger.critical('Available network interfaces:')

        for name, addr in net_ifs.items():
            logger.critical(f'\t{name}: {addr}')
        exit(1)

    if args.network_interface not in net_ifs:
        logger.critical(
            f'Network Interface {args.network_interface} is not found')
        logger.critical('Available network interfaces:')

        for name, addr in net_ifs.items():
            logger.critical(f'\t{name}: {addr}')
        exit(1)

    if args.debug:
        reset_logger_level(logger, logging.DEBUG)

    arguments = args
    arguments.ip = net_ifs[args.network_interface][0]
    arguments.netmask = net_ifs[args.network_interface][1]

    t = threading.Thread(target=worker_connector, args=(
        arguments.ip, arguments.netmask, arguments.port))
    t.start()

    consumer_process = Process(target=consumer, args=(working_queue, ))
    consumer_process.start()

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
    working_queue.put('exit')
    consumer_process.join()
    t.join()
