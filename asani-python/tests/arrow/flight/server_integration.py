from typing import List

from pydantic import BaseModel

from asani.arrow.flight.flight_processor import FlightProcessor
from asani.arrow.flight.server import AsaniFlightServer


class Frame(BaseModel):
    streamName: str
    image: bytes


class Detection(BaseModel):
    stream: str
    label: str
    score: float
    bbox: list[float]


class ObjectTrackingProcessor(FlightProcessor[Frame, Detection]):
    def __init__(self):
        super().__init__(request_model=Frame, response_model=Detection)

    def command(self) -> str:
        return "object_tracking"

    def process(self, frames: List[Frame]) -> List[Detection]:
        detections = []
        for frame in frames:
            detections.append(
                Detection(
                    stream=frame.streamName,
                    label="person 1",
                    score=57.0,
                    bbox=[12, 31.2, 54.1, 45.2],
                )
            )
        return detections


if __name__ == "__main__":
    location = "grpc://localhost:8815"

    tracking_processor = ObjectTrackingProcessor()

    server = AsaniFlightServer(location=location, processors=[tracking_processor])

    server.serve()
