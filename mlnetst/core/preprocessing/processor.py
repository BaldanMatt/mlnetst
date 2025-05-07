import os

from mlnetst.core.preprocessing.manager import PipelineStep
from typing import Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess

import pandas as pd

class Processor(PipelineStep):
    def __init__(self,
                 name: str,
                 data_technology: str,
                 filter_method: str,
                 norm_method: str,
                 output_path: Path = None) -> None:
        super().__init__(name)
        self.data_technology = data_technology
        self.filter_method = filter_method
        self.norm_method = norm_method
        self.output_path = output_path
        self._processed_data: Optional[Any] = None

    @property
    def processed_data(self) -> Any:
        if self._processed_data is None:
            raise ValueError("Processed data has not been generated yet.")
        return self._processed_data

    def is_already_processed(self) -> bool:
        return self.output_path is not None and self.output_path.exists()

    def run(self) -> None:
        if self.output_path and self.output_path.exists():
            # Load from cache if available
            self._processed_data = self.load_processed_data()
        else:
            # Get input data from dependency
            input_data = self.get_input_from(self.dependencies[0].name, key="data")
            self._processed_data = self.process(input_data)
            if self.output_path:
                self.save_processed_data()

        self.set_output("processed_data", self._processed_data)

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        pass

    @abstractmethod
    def load_processed_data(self) -> Any:
        pass

    @abstractmethod
    def save_processed_data(self) -> None:
        """Save processed data to file"""
        pass

    def __str__(self) -> str:
        return f"Processor: {self.name} | Technology: {self.data_technology}\n\t\t- Filter: {self.filter_method}\n\t\t- Normalization: {self.norm_method}\n\t\t- Output Path: {self.output_path}\n\t\tDependencies: {[dep.name for dep in self.dependencies]}"

class SnDataProcessor(Processor):
    proj_dir = Path(__file__).resolve().parents[3]
    data_dir = proj_dir / "data"
    tmp_dir = data_dir / "tmp"
    def __init__(self,
                 name: str,
                 filter_method: str,
                 norm_method: str,
                 output_path: Optional[Path] = None) -> None:
        super().__init__(
            name="snRNA Data Processor",
            data_technology="snrna",
            filter_method=filter_method,
            norm_method=norm_method,
            output_path=output_path
        )

    def process(self, input_data: Any) -> Any:
        # Annotate
        ## Annotate genes
        genes_str = ";".join(input_data.var_names) + "\n"

        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        with open(self.tmp_dir / "genes.txt", "w") as f:
            f.write(genes_str)
        subprocess.run(["Rscript", str(self.proj_dir / "mlnetst/utils/annotate_genes.R"),
                        "--input_file_path="+str(self.tmp_dir / "genes.txt"),
                        "--output_file_path="+str(self.tmp_dir / "genes_annotated.csv")],
                       cwd=self.proj_dir
                       )
        annotated_genes_df = pd.read_csv(self.tmp_dir / "genes_annotated.csv",
                                         sep=";",
                                         )
        print(annotated_genes_df)
        ## Annotate cells

        # # Filter

        # Transform
        ## Normalization
        ## Log transformation
        ## Scaling

        # Embedding
        ## PCA

        processed = input_data
        return processed

    def load_processed_data(self) -> Any:
        pass

    def save_processed_data(self) -> None:
        pass

class MerscopeDataProcessor(Processor):
    def __init__(self,
                 name: str,
                 filter_method: str,
                 norm_method: str,
                 output_path: Optional[Path] = None) -> None:
        super().__init__(
            name="merscope Data Processor",
            data_technology="merscope",
            filter_method=filter_method,
            norm_method=norm_method,
            output_path=output_path
        )
    def process(self, input_data: Any) -> Any:
        processed = input_data
        return processed
    def load_processed_data(self) -> Any:
        pass
    def save_processed_data(self) -> None:
        pass

class ProcessorFactory:
    @staticmethod
    def produce_processor(name: str,
                          data_technology: str,
                          filter_method: str,
                          norm_method: str,
                          output_path: Optional[Path] = None) -> Processor:
        if data_technology == "snrna":
            return SnDataProcessor(name, filter_method, norm_method, output_path)
        elif data_technology == "merscope":
            return MerscopeDataProcessor(name, filter_method, norm_method, output_path)
        else:
            raise ValueError(f"Unknown data technology: {data_technology}")

