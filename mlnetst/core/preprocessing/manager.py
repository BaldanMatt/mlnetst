from abc import ABC, abstractmethod
from typing import Any


class Builder(ABC):
    @property
    @abstractmethod
    def pipeline(self) -> Any:
        pass

    @abstractmethod
    def produce_loader(self) -> Any:
        pass

    @abstractmethod
    def produce_preprocessor(self) -> Any:
        pass

    @abstractmethod
    def produce_integrator(self) -> Any:
        pass

    @abstractmethod
    def produce_embedder(self) -> Any:
        pass

class PipelineStep():
    pass

class Pipeline():

    def __init__(self) -> None:
        self.steps = []

    def add_step(self, step: PipelineStep) -> None:
        self.steps.append(step)
