from typing import Optional, Any
import anndata
from pathlib import Path
from abc import ABC, abstractmethod
from mlnetst.core.preprocessing.manager import PipelineStep
from mlnetst.core.preprocessing.processor import RANDOM_STATE
import decoupler as dc


class Embedder(PipelineStep):
    def __init__(self, name: str,
                 output_path: Path = None,
                 force: bool = False) -> None:
        super().__init__(name)
        self.output_path = output_path
        self.force = force
        self._embedded_data: Optional[Any] = None
    @property
    def embedded_data(self) -> Any:
        if self._embedded_data is None:
            raise ValueError("Embedded data has not been generated yet.")
        return self._embedded_data

    def is_alread_embedded(self) -> bool:
        return self.output_path is not None and self.output_path.exists()

    def run(self) -> None:
        if self.is_alread_embedded() and not self.force:
            self._embedded_data = self.load_embedded_data()
        else:
            x_hat_s = self.get_input_from(self.dependencies[0].name, key="integrated_data")
            self._embedded_data = self.embed(x_hat_s)
            if self.output_path:
                self.save_embedded_data()
        self.set_output("embedded_data", self._embedded_data)

    @abstractmethod
    def embed(self, x_hat_s: anndata.AnnData) -> anndata.AnnData:
        pass
    @abstractmethod
    def load_embedded_data(self) -> anndata.AnnData:
        pass
    @abstractmethod
    def save_embedded_data(self) -> None:
        pass

class DecouplerEmbedder(Embedder):
    def __init__(self,
                 name:str,
                 output_path: Path = None,
                 force: bool = False,
                 embedding_kws: Optional[dict[str, str | float | int]] = None) -> None:
        super().__init__(name, output_path, force)
        default_kws = {
            'n_components': 2,
            'n_neighbors': 15,
            'random_state': RANDOM_STATE,
        }
        self.embedding_kws = default_kws if embedding_kws is None else {**default_kws, **embedding_kws}

    def embed(self, x_hat_s: anndata.AnnData) -> anndata.AnnData:

        return x_hat_s
    def load_embedded_data(self) -> anndata.AnnData:
        return anndata.read_h5ad(self.output_path)
    def save_embedded_data(self) -> None:
        self._embedded_data.write_h5ad(self.output_path)

class EmbedderFactory:
    @staticmethod
    def produce_embedder(name: str,
                         embedding_method: str,
                         output_path: Path,
                         **kwargs) -> Embedder:
        if embedding_method == "decoupler":
            return DecouplerEmbedder(name, output_path, **kwargs)
        else:
            raise ValueError("Embedder is not supported")