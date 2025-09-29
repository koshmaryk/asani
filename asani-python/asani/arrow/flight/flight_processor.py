from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Type
from pydantic import BaseModel
from asani.arrow.serializer import Serializer

In = TypeVar("In", bound=BaseModel)
Out = TypeVar("Out", bound=BaseModel)


class FlightProcessor(ABC, Generic[In, Out]):
    def __init__(self, request_model: Type[In], response_model: Type[Out]):
        super().__init__()
        self.request_serializer = Serializer(request_model)
        self.response_serializer = Serializer(response_model)

    async def process_request(self, reader, writer):
        models = []
        table = reader.read_all()

        try:
            models = self.request_serializer.from_table(table)
        except Exception as e:
            print(f"Error parsing records to model: {table}, error: {e}")

        result = self.process(models)
        response = self.response_serializer.to_table(result)

        writer.begin(response.schema)
        writer.write_table(response)

    @abstractmethod
    def command(self) -> str:
        raise NotImplemented()

    @abstractmethod
    def process(self, models: List[In]) -> List[Out]:
        raise NotImplemented()
