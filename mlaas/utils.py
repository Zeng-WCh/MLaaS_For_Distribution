import psutil
import logging
import json
import socket
import struct

from typing import Tuple, Dict, TextIO


def get_broadcast_address(ip: str, netmask: str) -> str:
    ip = struct.unpack('!I', socket.inet_aton(ip))[0]
    netmask = struct.unpack('!I', socket.inet_aton(netmask))[0]
    # print(f'IP: {ip}, Netmask: {netmask}')

    broadcast_addr = socket.inet_ntoa(
        struct.pack('!I', ip | (~netmask & 0xFFFFFFFF)))
    # print(f'Broadcast: {broadcast_addr}')
    return broadcast_addr


def str_to_dict(s: str) -> Dict:
    # print(s)
    if s.startswith('{') and s.endswith('}'):
        return json.loads(s)
    # split by = and strip the space
    return {k.strip(): v.strip() for k, v in [pair.split('=') for pair in s.split(',')]}


def get_network_interface() -> Dict[str, Tuple[str, str]]:
    net_ifs = psutil.net_if_addrs()
    ifnames = dict()

    for name, addrs in net_ifs.items():
        # ifnames[name] = addrs
        for addr in addrs:
            # IPv4 only
            if addr.family == 2:
                ifnames[name] = (addr.address, addr.netmask)

    return ifnames


def init_logger(logger_name: str, log_file: None | str = None, log_stream: None | TextIO = None) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    # Make the logger be able to handle all levels of logs
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_log = logging.FileHandler(log_file)
    console_log = logging.StreamHandler(log_stream)
    # Set the level of the handlers, so that the logger can handle different levels of logs
    console_log.setLevel(logging.INFO)
    file_log.setLevel(logging.INFO)
    console_log.setFormatter(formatter)
    file_log.setFormatter(formatter)
    logger.addHandler(console_log)
    logger.addHandler(file_log)
    return logger


def reset_logger_level(logger: logging.Logger, level: int):
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
