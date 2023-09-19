import nats
from .utils import _f

async def packet_handler(packet):
    subject = packet.subject
    reply = packet.reply
    data = packet.data.decode()
    _f("info", "Received a message on '{subject} {reply}': {data}".format(
        subject=subject, reply=reply, data=data)
    )

class Pole:
    def __init__(self, server):
        self.server = server
    async def on(self, frequency):
        nc = await nats.subscribe(f'nats://{self.server}')
        sub = await nc.subscribe(frequency)
        self.sub = sub
        self.nc = nc
        return self.nc
    async def off(self):
        await self.sub.unsubscribe()
        await self.nc.drain()
    async def receive(self, frequency):
        sub = await self.nc.subscribe(frequency, cb=packet_handler)