from typing import List
from pyarrow.flight import FlightServerBase
from asani.arrow.flight.flight_processor import FlightProcessor
import asyncio


class AsaniFlightServer(FlightServerBase):
    def __init__(self, location, processors: List[FlightProcessor], **kwargs):
        super().__init__(location, **kwargs)
        self.processors = processors
        print("Asani Server started.")

    def do_exchange(self, context, descriptor, reader, writer):
        command = descriptor.command.decode("utf-8")

        processor = next(
            (proc for proc in self.processors if proc.command() == command),
            None,  # Return None if no processor matches the command
        )

        if processor is not None:
            asyncio.run(processor.process_request(reader, writer))
        else:
            raise Exception(f"No registered processor for a command: {command}")
