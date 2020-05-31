from syft.workers.websocket_server import WebsocketServerWorker
import syft as sy
import torch
hook = sy.TorchHook(torch)
import threading
import asyncio
import _thread

server_list = []
port_zero = 8778
threads = []
def start_server(index):
    server = server_list[index]
    asyncio.set_event_loop(asyncio.new_event_loop())
    server.start()

for i in range(0,20):
    kwargs = {
        "id": str(i),
        "host": "localhost",
        "port": port_zero + i,
        "hook": hook,
        "verbose": True
    }
    server = WebsocketServerWorker(**kwargs)
    server_list.append(server)
    x = threading.Thread(target=start_server, args=(i,), daemon=True)
    x.start()

