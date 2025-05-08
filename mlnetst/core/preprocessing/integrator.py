from typing import Optional, Any
from pathlib import Path
from abc import ABC, abstractmethod

from mlnetst.core.preprocessing.manager import PipelineStep
from mlnetst.core.preprocessing.processor import MerscopeDataProcessor, SnDataProcessor


class Integrator(PipelineStep):
    def __init__(self, name: str,
                 output_path:  Path = None) -> None:
        super().__init__(name)
        self.output_path = output_path
        self._integrated_data: Optional[Any] = None
    @property
    def integrated_data(self) -> Any:
        if self._integrated_data is None:
            raise ValueError("Mapped data has not been generated yet.")
        return self._integrated_data

    def is_already_integrated(self) -> bool:
        return self.output_path is not None and self.output_path.exists()

    def run(self) -> None:
        if self.output_path and self.output_path.exists():
            self._integrated_data = self.load_integrated_data()
        else:
            # Get input data from dependency
            ## Check if we have right dependencies
            if len(self.dependencies) != 2:
                raise ValueError("Integrator requires two dependencies. snRNAProcessor and MerscopeProcessor.")
            if not all([isinstance(dep, SnDataProcessor) or isinstance(dep, MerscopeDataProcessor) for dep in self.dependencies]):
                raise ValueError("Integrator requires two dependencies, one of type SnDataProcessor and one of type MerscopeDataProcessor.")
            for dep in self.dependencies:
                # print("[DEBUG] type of dep: ", type(dep))
                if isinstance(dep, SnDataProcessor):
                    snrna_input_data = self.get_input_from(dep.name, key="processed_data")
                if isinstance(dep, MerscopeDataProcessor):
                    # print("[DEBUG] are we entering in extraction of merscope?")
                    merscope_input_data = self.get_input_from(dep.name, key="processed_data")
            self._integrated_data = self.integrate(snrna_input_data, merscope_input_data)

            if self.output_path:
                self.save_integrated_data()

        # print("[DEBUG] setting output with ", self._integrated_data,)
        self.set_output("integrated_data", self._integrated_data)

    @abstractmethod
    def integrate(self, snrna_input_data: Any, merscope_input_data: Any) -> Any:
        pass

    @abstractmethod
    def load_integrated_data(self) -> None:
        pass
    @abstractmethod
    def save_integrated_data(self) -> None:
        pass

    def __str__(self):
        return f"Integrator: {self.name}\n\t\t- Output path: {self.output_path}"

class TangramMapper(Integrator):
    def __init__(self, name: str,
                 output_path: Path = None,
                 integration_kws: Optional[dict] = {
                     'point_of_view': 'slices',
                     'which_one': "mouse1_slice153",
                 }) -> None:
        super().__init__(name, output_path)
        self.integration_kws = integration_kws

    def integrate(self, snrna_input_data: Any, merscope_input_data: Any) -> Any:
        integrated = merscope_input_data
        if self.integration_kws.get('point_of_view') == 'slices':

            if self.integration_kws.get('which_one') == 'all':
                ...
            else:
                spdata = merscope_input_data.filter_by_coordinate_system(self.integration_kws.get('which_one'))
        else:
            ...
            if self.integration_kws.get('which_one') == 'all':
                ...
            else:
                spdata = merscope_input_data.filter_by_coordinate_system(self.integration_kws.get('which_one'))

        # print("[DEBUG intern] what is integrated: ", integrated)
        return spdata

    def load_integrated_data(self) -> None:
        pass

    def save_integrated_data(self) -> None:
        pass

class IntegratorFactory:
    @staticmethod
    def produce_integrator(name: str,
                           integration_method: str,
                           output_path: Path,
                           **kwargs) -> Integrator:
        if integration_method == "tangram":
            return TangramMapper(name, output_path, **kwargs)
        else:
            raise ValueError("Integrator is not supported")
