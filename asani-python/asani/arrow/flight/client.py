from typing import TypeVar, Generic, Type, List
from pyarrow.flight import FlightClient, FlightDescriptor
from pydantic import BaseModel

from asani.arrow.serializer import Serializer

Req = TypeVar("Req", bound=BaseModel)
Resp = TypeVar("Resp", bound=BaseModel)


class AsaniFlightClient(Generic[Req, Resp]):
    def __init__(
        self, host: str, port: int, request_model: Type[Req], response_model: Type[Resp]
    ):
        self.client = FlightClient((host, port))
        self.request_serializer = Serializer(request_model)
        self.response_serializer = Serializer(response_model)

    async def call(self, command: str, request_data: List[Req]) -> List[Resp]:
        descriptor = FlightDescriptor.for_command(command)
        writer, reader = self.client.do_exchange(descriptor)

        root = self.request_serializer.to_table(request_data)

        writer.begin(root.schema)
        writer.write_table(root)
        writer.done_writing()

        # Read response from server
        response_table = reader.read_all()
        response = self.response_serializer.from_table(response_table)

        return response
