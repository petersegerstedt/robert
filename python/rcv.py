import signal
import time

import socket
import struct
from datetime import datetime

import sys

UDP_IP = "224.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

def handler(signum, frame):
    res = input("Ctrl-c was pressed. Do you really want to exit? y/n ")
    if res == 'y':
        sock.close()
        sys.exit(0)

signal.signal(signal.SIGINT, handler)

def on_time(data):
    n, = struct.unpack_from('I', data)
    if 8 != n:
        return
    ts, = struct.unpack('d', data[8:])
    print(ts, datetime.utcfromtimestamp(ts))
while True:
    buf, addr = sock.recvfrom(512) # buffer size is 512 bytes
    cmd, data = buf[:4].decode('ascii'), buf[4:]

    if 'TIME' == cmd:
        on_time(data[4:])
    time.sleep(0.1)
