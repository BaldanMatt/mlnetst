import anndata
import spatialdata
from mlnetst.core.preprocessing.manager import PipelineStep
from typing import Any, Optional, List
from abc import ABC, abstractmethod
from pathlib import Path

class DataLoader(PipelineStep):
    def __init__(self, name: str, data_technology: str, file_path: Path = None) -> None:
        super().__init__(name)
        self.data_technology = data_technology
        self.file_path = Path(file_path)
        self._data: Optional[Any] = None

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    @property
    def data(self) -> Any:
        if self._data is None:
            raise ValueError("Data has not been loaded yet.")
        return self._data

    def run(self) -> Any:
        loaded_data = self.load()
        print("[DEBUG]", loaded_data)
        self.set_output("data", loaded_data)

    @abstractmethod
    def load(self) -> None:
        pass

    def __str__(self) -> str:
        return f"Loader: {self.name} | Technology: {self.data_technology} | Path: {self.file_path}"


class SnDataLoader(DataLoader):
    def __init__(self, name, file_path: Path):
        super().__init__(
            name="snRNA Data Loader",
            data_technology="snrna",
            file_path=file_path
        )

    def load(self) -> anndata.AnnData:
        # Load snRNA data
        return anndata.read_h5ad(self.file_path)


class MerscopeDataLoader(DataLoader):
    def __init__(self, name, file_path: Path):
        super().__init__(
            name="Merscope Data Loader",
            data_technology="merscope",
            file_path=file_path
        )

    def load(self) -> None:
        # Load Merscope data
        self._data = spatialdata.read_zarr(self.file_path)
        print(self.data)

class DataLoaderFactory:
    @staticmethod
    def produce_loader(name:str, data_technology: str, file_path: Path) -> DataLoader:
        if data_technology == "snrna":
            return SnDataLoader(name, file_path)
        elif data_technology == "merfish" or data_technology == "merscope":
            return MerscopeDataLoader(name, file_path)
        else:
            raise ValueError("Data Technology is not supported")
