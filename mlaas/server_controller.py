import argparse
import asyncio
import logging
import json
import sys
import subprocess
import socket
import schedule
import os
import time
import threading

from aiohttp import ClientSession
from config import DB_CONFIG, MODEL_DIRECTORY, CONNECTION_TIMEOUT, BROADCAST_PORT
from db_connector import DatabaseConnector, Dataline
from flask import Flask, request, jsonify
from argparse import ArgumentParser
from typing import Dict, Tuple, Any
from utils import get_network_interface, init_logger, reset_logger_level, get_broadcast_address
from multiprocessing import Process, Queue, Lock
from model_config import GENERAL_REQUIRED_PARAMS, SPECIAL_REQUIRED_PARAMS, ALL_MODEL_CLASSES


class WorkerInfo(object):
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.last_update = time.time()

    def update(self, ip: str | None = None, port: int | None = None):
        if ip is not None:
            self.ip = ip
        if port is not None:
            self.port = port
        self.last_update = time.time()

    def is_alive(self):
        return time.time() - self.last_update < CONNECTION_TIMEOUT

    def get_ip(self):
        return self.ip

    def get_port(self):
        return self.port


app = Flask('server_controller')
worker_config: Dict[Tuple[str, int], WorkerInfo] = dict()
worker_lock = Lock()
arguments: None | argparse.Namespace = None
db: None | DatabaseConnector = None

request_queue = Queue()

# logging.basicConfig(level=logging.DEBUG)
logger: logging.Logger = init_logger(
    'Server Controller', log_file='server_controller.log', log_stream=sys.stdout)


# def get_client_config(config_file):
#     with open(config_file, 'r') as f:
#         config_json = json.load(f)
#     # print(config_json)
#     global worker_config
#     worker_config = config_json


def send_broadcast(broadcast_addr: str, port: int, message: str):
    logger.debug(f'Sending broadcast to {broadcast_addr}:{port}')
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.sendto(message.encode(), (broadcast_addr, port))
    sock.close()


def check_worker_status(lock):
    lock.acquire()

    for sock in worker_config:
        if not worker_config[sock].is_alive():
            logger.info(
                f'Worker {sock[0]}:{sock[1]} can not response to heard beat, disconnected...')
            worker_config.pop(sock)
        else:
            logger.info(
                f'Worker {sock[0]}:{sock[1]} response to heard beat, alive')

    lock.release()


def start_autodetection(server_ip: str, netmask: str):
    broadcast_addr = get_broadcast_address(server_ip, netmask)

    package = {
        'type': 'worker_discovery',
        'ip': server_ip,
        'port': arguments.port,
        'endpoint': '/api/worker_discovery',
    }

    payload = json.dumps(package)

    send_broadcast(broadcast_addr, BROADCAST_PORT, payload)
    # check_worker_status(lock)


def auto_detection_daemon(server_ip, netmask, lock):
    schedule.every(5).seconds.do(start_autodetection, server_ip, netmask)
    schedule.every(CONNECTION_TIMEOUT).seconds.do(
        check_worker_status, lock)

    while True:
        schedule.run_pending()
        time.sleep(1)


def cunsumer(msgq: Queue, db: DatabaseConnector):
    global worker_config, arguments
    env = os.environ.copy()
    env['GLOO_SOCKET_IFNAME'] = arguments.network_interface
    env['TP_SOCKET_IFNAME'] = arguments.network_interface
    while True:
        msg = msgq.get()
        logger.debug(f'Get data: {msg}')
        if msg == 'exit':
            break
        msg, local_worker_config = msg
        training_id = msg['training_id']
        logger.info(f'Updating training record{training_id}...')
        query = f'UPDATE {DB_CONFIG["table"]} SET training_status="running", completed_at=NOW(), model_location="{msg["model_export_name"]}", model_name="{msg["model_file_name"]}" WHERE id = {training_id};'
        param: dict = msg['params']
        logger.debug(f'Query: {query}')
        success, cursor = db.execute_query(query, False)
        if success:
            logger.info(f'Update training record {training_id} successfully')
        else:
            logger.error(f'Update training record {training_id} failed')

        if cursor is not None:
            cursor.close()

        param_str = json.dumps(param)
        # worker_lock.acquire()

        logger.debug(
            f'Command: python3 server.py --world_size {len(local_worker_config) + 1} --port {arguments.rpc_port} --model {msg["model"]} --params {param_str}')

        p = subprocess.Popen(
            [
                'python3', 'server.py',
                '--world_size', str(len(local_worker_config) + 1),
                '--port', str(arguments.rpc_port),
                '--model', msg['model'],
                '--model_export_name', msg["model_export_name"],
                '--params', param_str,
                # '--debug'
            ],
            env=env
        )

        # worker_lock.release()
        p.wait()

        # Update DB
        training_id = msg['training_id']
        logger.info(f'Updating training record{training_id}...')
        query = f'UPDATE {DB_CONFIG["table"]} SET training_status="completed", completed_at=NOW(), model_location="{msg["model_export_name"]}", model_name="{msg["model_file_name"]}" WHERE id = {training_id};'
        logger.debug(f'Query: {query}')
        success, cursor = db.execute_query(query, False)
        if success:
            logger.info(f'Update training record {training_id} successfully')
        else:
            logger.error(f'Update training record {training_id} failed')

        if cursor is not None:
            cursor.close()


def create_record(data: Dict[str, Any]):
    user_id = data['id']

    line = Dataline(data['model'], created_by=user_id)
    sql_query = f'INSERT INTO {DB_CONFIG["table"]} (submit_date, {line.line_description()}) VALUES (NOW(), {line.create_placeholder()});'
    sql_args = line.to_sql_arguments()

    global db

    if db is None:
        logger.error(f'Database is not initialized')
        return

    success, cursor = db.execute_query(sql_query, True, *sql_args)

    if success:
        id = cursor.lastrowid
        if cursor is not None:
            cursor.close()
        logger.info(f'Insert record successfully')
        logger.debug(f'Last row id: {id}')
        data['training_id'] = id
    else:
        logger.error(f'Insert record failed')
        if cursor is not None:
            cursor.close()

    model_export_path = os.path.join(
        MODEL_DIRECTORY, data['model_export_name'])

    model_export_path = os.path.abspath(model_export_path)
    data['model_export_name'] = model_export_path
    data['model_file_name'] = os.path.basename(model_export_path)

    return data


async def send_post_request(url, payload, json=False):
    if json:
        async with ClientSession() as session:
            async with session.post(url, json=payload) as response:
                return response.status
    else:
        async with ClientSession() as session:
            async with session.post(url, data=payload) as response:
                return response.status


@app.route('/api/worker_discovery', methods=['POST'])
def worker_register():
    if request.method != 'POST':
        return 'Invalid method'

    json_data = request.get_json()
    logger.debug(f'Receive from ip: {request.remote_addr}')
    logger.debug(f'JSON Data: {json_data}')
    ip = json_data['ip']
    port = json_data['port']
    worker_lock.acquire()
    if (ip, port) not in worker_config:
        worker_config[(ip, port)] = WorkerInfo(ip, port)
        logger.info(f'Worker {ip}:{port} is registered')
    else:
        worker_config[(ip, port)].update(ip, port)
        logger.info(f'Worker {ip}:{port} response to heard beat')
    worker_lock.release()
    return 'OK'


@app.route('/api/start', methods=['POST'])
async def start():
    if request.method != 'POST':
        return 'Invalid method'

    json_data = request.get_json()
    logger.debug(f'Get data: {json_data}')
    json_data = create_record(json_data)

    logger.info(f'Get command from {request.remote_addr}')

    global worker_config, arguments, worker_lock

    world_size = len(worker_config) + 1

    local_worker_config = dict()

    # Get a worker sets
    worker_lock.acquire()
    for socket_info in worker_config:
        if worker_config[socket_info].is_alive():
            local_worker_config[socket_info] = worker_config[socket_info]
    worker_lock.release()

    for i, socket_info in enumerate(local_worker_config):
        logger.info(f'Sending file to worker {i+1}...')
        worker = local_worker_config[socket_info]

        worker_address = f'http://{worker.get_ip()}:{worker.get_port()}/api/upload'
        filename = json_data['file']

        if filename and filename != '':
            with open(filename, 'rb') as file:
                send_name = os.path.basename(filename)
                status = await send_post_request(
                    worker_address,  {'file': file, 'filename': send_name})
                if status != 200:
                    logger.error(
                        f'Error: Worker {i+1} failed to receive file, status code: {status}')
                    return 'Error'

    result = list()

    for i, socket_info in enumerate(local_worker_config):
        worker = local_worker_config[socket_info]
        worker_address = f'http://{worker.get_ip()}:{worker.get_port()}/api/create_worker'
        payload = {
            'rank': str(i+1),
            'world_size': str(world_size),
            'server_ip': arguments.server_ip,
            'server_rpc_port': str(arguments.rpc_port),
            'filetype': json_data['filetype'],
            'filename': send_name if filename and filename != '' else "",
            'model': json_data['model'],
            'model_export_name': json_data['model_export_name'],
            'params': json_data['params'],
        }
        logger.info(f'Sending command to {worker_address}')
        logger.info(f'Payload: {payload}')
        # threads.append(asyncio.create_task(send_post_request(worker_address, payload)))
        result.append(asyncio.create_task(send_post_request(
            worker_address, payload, True)))
    await asyncio.wait(result, return_when=asyncio.ALL_COMPLETED)
    # Check the status of the threads
    for i, status in enumerate(result):
        # print(status.result())
        if status.result() != 200:
            logger.error(
                f'Error: Worker {i+1} failed to start, status code: {status}')
            return 'Error'

    data = (json_data, local_worker_config)
    request_queue.put(data)
    # request_queue.put('start')
    return 'OK'


@app.route('/api/models', methods=['GET', 'POST'])
def get_all_model_info():
    return jsonify(ALL_MODEL_CLASSES)


@app.route('/api/required_params', methods=['GET', 'POST'])
def get_required_params():
    model_name = request.args.get('model')
    if model_name not in ALL_MODEL_CLASSES:
        return 'Invalid model name', 400
    return jsonify({**GENERAL_REQUIRED_PARAMS, **SPECIAL_REQUIRED_PARAMS[model_name]})


if __name__ == '__main__':
    parser = ArgumentParser('Parameter Server Controller')

    parser.add_argument('--port', type=int, default=8000,
                        help='Controller API Port')
    parser.add_argument('--rpc_port', type=int, default=29500, help='RPC Port')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    # parser.add_argument('--client_config', type=str,
    #                     required=True, help='Client configuration file in json')
    parser.add_argument('--network_interface', type=str,
                        default=None, help='RPC Network Interface')
    # parser.add_argument('--upload_dir', type=str,
    #                     default='./uploads', help='Upload directory')

    args = parser.parse_args()

    net_ifs = get_network_interface()
    if args.network_interface is None or args.network_interface == '':
        logger.critical('Network Interface is not specified')
        logger.critical('Available network interfaces:')

        for name, addr in net_ifs.items():
            logger.critical(f'\t{name}: {addr}')
        sys.exit(1)

    if args.network_interface not in net_ifs:
        logger.critical(
            f'Network Interface {args.network_interface} is not found')
        logger.critical('Available network interfaces:')

        for name, addr in net_ifs.items():
            logger.critical(f'\t{name}: {addr}')
        sys.exit(1)

    if args.debug:
        reset_logger_level(logger, logging.DEBUG)

    try:
        db = DatabaseConnector(**DB_CONFIG)
    except Exception as e:
        logger.error(f'Failed to connect to database: {e}')
        sys.exit(1)

    logger.info(f'Setting model export directory to {MODEL_DIRECTORY}')

    if not os.path.exists(MODEL_DIRECTORY):
        os.makedirs(MODEL_DIRECTORY)

    arguments = args
    # Add IPv4
    arguments.server_ip = net_ifs[args.network_interface][0].strip()
    # get_client_config(args.client_config)
    # print(config)
    # timed_threads = start_autodetection(arguments.server_ip,
    #                                     net_ifs[args.network_interface][1].strip())

    # checker = Process(target=start_autodetection, args=(
    #     arguments.server_ip, net_ifs[args.network_interface][1].strip(), worker_lock,))
    ps_server = Process(target=cunsumer, args=(request_queue, db, ))
    ps_server.start()

    p = Process(target=auto_detection_daemon, args=(
        arguments.server_ip, net_ifs[args.network_interface][1].strip(), worker_lock, ))
    p.start()
    # checker.start()
    logger.info(f'Using {args.network_interface} as RPC interface')
    logger.info(f'Controller API is running on port {args.port}')
    # logger.info(f'Upload directory: {args.upload_dir}')
    # app.config['UPLOAD_FOLDER'] = args.upload_dir
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
    db.close_db()
    request_queue.put('exit')
    ps_server.join()
    p.join()
    # checker.join()
