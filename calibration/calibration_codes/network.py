from enum import Enum

TCP_IP = '127.0.0.1'
TCP_PORT = 6031
BUFFER_SIZE = 1024


class PacketType(Enum):
    show_calib_image = 1


class Packet:
    def __init__(self, type=PacketType.show_calib_image, content=0):
        self.type = type
        self.content = content

