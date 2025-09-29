from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic

from pydantic import BaseModel

In = TypeVar("In", bound=BaseModel)
Out = TypeVar("Out", bound=BaseModel)


class Processor(ABC, Generic[In, Out]):
    @abstractmethod
    def command(self) -> str:
        raise NotImplemented()

    @abstractmethod
    def process(self, data: List[In]) -> List[Out]:
        raise NotImplemented()
