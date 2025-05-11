from typing import Optional, Any
from pathlib import Path
from abc import ABC, abstractmethod
import time

import anndata
import pandas as pd
import numpy as np
import tangram as tg

from mlnetst.core.preprocessing.manager import PipelineStep
from mlnetst.core.preprocessing.processor import MerscopeDataProcessor, SnDataProcessor, RANDOM_STATE


class Integrator(PipelineStep):
    def __init__(self, name: str,
                 output_path:  Path,
                 force: bool) -> None:
        super().__init__(name)
        self.output_path = output_path
        self.force = force
        self._integrated_data: Optional[Any] = None
    @property
    def integrated_data(self) -> Any:
        if self._integrated_data is None:
            raise ValueError("Mapped data has not been generated yet.")
        return self._integrated_data

    def is_already_integrated(self) -> bool:
        return self.output_path is not None and self.output_path.exists()

    def run(self) -> None:
        if self.is_already_integrated() and not self.force:
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
    def integrate(self, snrna_input_data: Any, merscope_input_data: Any) -> anndata.AnnData | None:
        pass

    @abstractmethod
    def load_integrated_data(self) -> anndata.AnnData | None:
        pass
    @abstractmethod
    def save_integrated_data(self) -> None:
        pass

    def __str__(self):
        return f"Integrator: {self.name}\n\t\t- Output path: {self.output_path}"

class TangramMapper(Integrator):
    def __init__(self, name: str,
                 output_path: Path = None,
                 force: bool = False,
                 integration_kws: Optional[dict[str, str | float | int]] = None) -> None:
        super().__init__(name, output_path, force)
        default_kws = {
            'point_of_view': 'slices',
            'which_one': "mouse1_slice153",
            'subset_scale_factor': 2.0,
            'num_epochs': 1000,
            'random_state': RANDOM_STATE,
        }
        self.integration_kws = default_kws if integration_kws is None else {**default_kws, **integration_kws}

    def integrate(self, x_c: Any, x_s: Any) -> Any:
        # Set default subset scale factor
        n_c, n_g = x_c.shape
        n_s, n_g_first = x_s.tables["table"].shape
        subset_scale_factor = float(self.integration_kws.get('subset_scale_factor'))
        # Filter data based on point of view and selection
        point_of_view = self.integration_kws.get('point_of_view')
        which_one = self.integration_kws.get('which_one')
        # tangram utils values
        num_epochs = self.integration_kws.get('num_epochs')

        if point_of_view == 'slices':
            if which_one != 'all':
                x_s = x_s.filter_by_coordinate_system(which_one)
        else:
            if which_one != 'all':
                x_s = x_s.filter_by_coordinate_system(which_one)

        x_s = x_s.tables["table"]
        # print("[DEBUG intern] what is integrated: ", integrated)

        # prepare tangram
        ## Subsample a number of cells from snrna_input_data to reduce the memory burden
        print(x_c.obs.columns)
        snrna_class_cat = x_c.obs["Allen.class_label"].value_counts()
        spdata_call_cat = x_s.obs["class_label"].value_counts()
        print(snrna_class_cat, spdata_call_cat)
        # Let's sample from snrna_input_data as much as X times the number of observations of spdata but using the
        # abundance of snrna_input_data Allen.class_label key observations
        weights_cat = x_c.obs["Allen.class_label"].map(
            x_c.obs["Allen.class_label"].value_counts() / x_c.obs[
                "Allen.class_label"].value_counts().sum()
        )
        row_indexes = x_c.obs.sample(
            n=int(subset_scale_factor * x_s.shape[0]),
            replace=False,
            weights=weights_cat,
            random_state = RANDOM_STATE
        ).index
        x_c = x_c[row_indexes, :]
        print(
            f"[INTEGRATOR] Subsampled snRNA data from {n_c} to {x_c.shape[0]} which is (n_s:{x_s.shape[0]} x {subset_scale_factor}) cells"
        )
        print(x_c.obs["Allen.class_label"].value_counts())
        markers_df = x_c.var.copy()

        markers = list(markers_df.index)

        tg.pp_adatas(x_c, x_s, genes = markers)

        # Run integration
        tic = time.time()
        mapping_matrix = tg.map_cells_to_space(
            x_c, x_s,
            num_epochs=num_epochs,
            mode="cells",
            density_prior="uniform",
            target_count = x_s.shape[0],
            random_state=RANDOM_STATE
        )
        x_hat_s = tg.project_genes(mapping_matrix, x_c)
        # Transfer annotations of snrna seq
        tg.project_cell_annotations(mapping_matrix, x_s, annotation="Allen.class_label")
        # Get the column names (class labels) from tangram_ct_pred
        class_names = x_s.obsm["tangram_ct_pred"].columns
        # Get index of highest probability for each row
        class_indices = np.argmax(x_s.obsm["tangram_ct_pred"].values, axis=1)
        # Map indices to actual class names
        x_hat_s.obs["Allen.class_label"] = [class_names[i] for i in class_indices]
        # Do the same for subclass predictions
        tg.project_cell_annotations(mapping_matrix, x_s, annotation="Allen.subclass_label")
        subclass_names = x_s.obsm["tangram_ct_pred"].columns
        subclass_indices = np.argmax(x_s.obsm["tangram_ct_pred"].values, axis=1)
        x_hat_s.obs["Allen.subclass_label"] = [subclass_names[i] for i in subclass_indices]
        toc = time.time()
        print(f"[INTEGRATOR] training of tangram is done in {toc - tic:.2f} seconds")
        return x_hat_s

    def load_integrated_data(self) -> None:
        return anndata.read_h5ad(self.output_path)

    def save_integrated_data(self) -> None:
        self._integrated_data.write_h5ad(self.output_path)

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
