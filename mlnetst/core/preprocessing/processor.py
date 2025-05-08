import os
import time

from mlnetst.core.preprocessing.manager import PipelineStep
from typing import Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
from tqdm import tqdm

import pandas as pd
import numpy as np
import anndata
import scanpy as sc

RANDOM_STATE = 42

class Processor(PipelineStep):
    def __init__(self,
                 name: str,
                 data_technology: str,
                 filter_method: str,
                 norm_method: str,
                 output_path: Path = None,
                 filter_kws: dict = None,
                 norm_kws: dict = None) -> None:
        super().__init__(name)
        self.data_technology = data_technology
        self.filter_method = filter_method
        self.norm_method = norm_method
        self.output_path = output_path
        self.filter_kws = filter_kws
        self.norm_kws = norm_kws
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

        # print("[DEBUG in processor] what is processed data: ", self._processed_data)
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
                 output_path: Optional[Path] = None,
                 filter_method: str = "default",
                 norm_method: str = "scanpy",
                 filter_kws: Optional[dict] = None,
                 norm_kws: Optional[dict] = None,) -> None:
        super().__init__(
            name="snRNA Data Processor",
            data_technology="snrna",
            filter_method=filter_method,
            filter_kws = filter_kws,
            norm_method=norm_method,
            norm_kws = norm_kws,
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
                                         index_col = "SYMBOL")

        print(annotated_genes_df)
        input_data.var = input_data.var.join(annotated_genes_df, how="left")
        # annotate mithocondrial genes
        input_data.var["is_mito"] = (input_data.var["SEQNAME"] == "MT") | (input_data.var["SEQNAME"] == "mt")
        input_data.obs["percent_mito"] = np.sum(
            input_data[:, input_data.var["is_mito"]].X,
            axis=1
        ) / np.sum(input_data.X, axis=1) * 100
        ## Annotate cells

        # # Filter
        tic = time.time()
        if self.filter_method == "default" or self.filter_method == "scrublet":
            # Set default values
            low_th: int = 100
            high_th: int = 6000
            mito_th: float = 3
            min_cells_per_gene: int = 10
            min_genes_per_cell: int = 10
            scrublet_th: float = .25
            do_remove_coding: bool = True

            if self.filter_kws:
                low_th = self.filter_kws.get("low_th", low_th)
                high_th = self.filter_kws.get("high_th", high_th)
                mito_th = self.filter_kws.get("mito_th", mito_th)
                min_cells_per_gene = self.filter_kws.get("min_cells_per_gene", min_cells_per_gene)
                min_genes_per_cell = self.filter_kws.get("min_genes_per_cell", min_genes_per_cell)
                scrublet_th = self.filter_kws.get("scrublet_th", scrublet_th)
                do_remove_coding = self.filter_kws.get("do_remove_coding", do_remove_coding)

            ## Filter genes
            if do_remove_coding:
                before_shape = input_data.shape
                input_data = input_data[:, (input_data.var["GENEBIOTYPE"] == "protein_coding")]
                after_shape = input_data.shape
                print(f"[PROCESSOR] Removed {before_shape[1] - after_shape[1]} genes due to non-coding, now is ({after_shape})")

            before_shape = input_data.shape
            if isinstance(input_data.X, anndata._core.views.SparseCSCMatrixView):
                # For sparse arrays
                non_zero_counts = input_data.X.getnnz(axis=0)
            else:
                # For dense arrays
                non_zero_counts = (input_data.X != 0).sum(axis=0)
            keep_columns = non_zero_counts > min_genes_per_cell
            input_data = input_data[:, keep_columns]
            after_shape = input_data.shape
            print(f"[PROCESSOR] Removed {before_shape[1] - after_shape[1]} genes due to low unique expression, now is ({after_shape})")

            ## Filter cells
            before_shape = input_data.shape
            input_data = input_data[input_data.obs["percent_mito"] < mito_th, :]
            after_shape = input_data.shape
            print(f"[PROCESSOR] Removed {before_shape[0] - after_shape[0]} cells due to high mitochondrial expression, now is ({after_shape})")

            before_shape = input_data.shape
            input_data = input_data[input_data.obs["nGene"] > low_th, :]
            input_data = input_data[input_data.obs["nGene"] < high_th, :]
            after_shape = input_data.shape
            print(f"[PROCESSOR] Removed {before_shape[0] - after_shape[0]} cells with too few or too many features, now is ({after_shape})")

            ## Doublet detection
            if self.filter_method == "scrublet":
                before_shape = input_data.shape
                sc.pp.scrublet(input_data, random_state=RANDOM_STATE)
                #input_data = input_data[input_data.obs["scrublet_score"] < scrublet_th, :]
                input_data = input_data[input_data.obs["predicted_doublet"], :]
                print(f"[PROCESSOR] Removed {before_shape[0] - input_data.shape[0]} cells due to doublet detection, now is ({input_data.shape})")
        toc = time.time()
        print(f"[PROCESSOR] Filtered data in {toc - tic:.2f} seconds")

        # Transform
        # set defaults
        target_sum: int = 1e4
        if self.norm_kws:
            target_sum = self.norm_kws.get("target_sum", target_sum)
        tic = time.time()
        ## Normalization
        if self.norm_method == "scanpy":
            sc.pp.normalize_total(input_data, target_sum=target_sum)


        elif self.norm_method == "deseq2":
            from scipy.sparse import csr_matrix
            geometric_means = np.power(np.mean(np.log1p(input_data.X), axis=0), np.e)
            ratios = input_data.X / geometric_means
            ratios = csr_matrix
            row_medians = []
            for i in tqdm(range(ratios.shape[0])):
                nonzero_vals = ratios[i].data
                median_val = np.median(nonzero_vals) if nonzero_vals.size > 0 else 1
                row_medians.append(median_val)
            medians = np.array(row_medians)
            input_data.X = input_data / medians[:, None]

        elif self.norm_method == "scran":
            input_data.write_h5ad(self.tmp_dir / "snrna_data_to_scran.h5ad")
            subprocess.run(["Rscript", str(self.proj_dir / "mlnetst/utils/scran_normalization.R"),
                            "--input_file_path="+str(self.tmp_dir / "snrna_data_to_scran.h5ad"),
                            "--output_file_path="+str(self.tmp_dir / "snrna_data_scran.h5ad")],
                           cwd=self.proj_dir
                           )
            input_data = anndata.read_h5ad(self.tmp_dir / "snrna_data_scran.h5ad")

        ## Log transformation
        sc.pp.log1p(input_data)
        input_data.X = input_data.X.tocsr()

        toc = time.time()
        print(f"[PROCESSOR] Normalized and transformed data in {toc - tic:.2f} seconds")

        ## Scaling

        processed = input_data
        return processed

    def load_processed_data(self) -> Any:
        return anndata.read_h5ad(self.output_path)

    def save_processed_data(self) -> None:
        self._processed_data.write_h5ad(self.output_path)

class MerscopeDataProcessor(Processor):
    def __init__(self,
                 name: str,
                 output_path: Optional[Path] = None,
                 filter_method: str = "default",
                 norm_method: str = "scanpy",
                 filter_kws: Optional[dict] = None,
                 norm_kws: Optional[dict] = None) -> None:
        super().__init__(
            name="merscope Data Processor",
            data_technology="merscope",
            filter_method=filter_method,
            norm_method=norm_method,
            output_path=output_path,
            filter_kws=filter_kws,
            norm_kws=norm_kws
        )
    def process(self, input_data: Any) -> Any:
        input_data.tables["table"].obs["sample_id"] = input_data.tables["table"].obs["sample_id"].astype("category")
        input_data.tables["table"].obs["slice_id"] = input_data.tables["table"].obs["slice_id"].astype("category")

        samples_categories = input_data.tables["table"].obs["sample_id"].cat.categories
        print("[DEBUG] samples categories: ", samples_categories)
        slices_categories = input_data.tables["table"].obs["slice_id"].cat.categories

        from spatialdata.transformations import get_transformation, set_transformation, Identity
        for sample_id in samples_categories:
            # print("[DEBUG] sampleid: ", sample_id)
            # Get unique slice_ids for current sample_id using groupby + unique
            slices_id_in_sample_id = input_data.tables["table"].obs.groupby('sample_id')['slice_id'].unique()[sample_id]
            # print("[DEBUG] slices in sampleid: ", slices_id_in_sample_id)
            for slice_id in slices_id_in_sample_id:
                set_transformation(input_data.shapes[slice_id], Identity(), to_coordinate_system=sample_id)
                set_transformation(input_data.shapes[slice_id], Identity(), to_coordinate_system=slice_id)
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
                          output_path: Optional[Path] = None,
                          **kwargs) -> Processor:
        if data_technology == "snrna":
            return SnDataProcessor(name, output_path, **kwargs)
        elif data_technology == "merscope":
            return MerscopeDataProcessor(name, output_path, **kwargs)
        else:
            raise ValueError(f"Unknown data technology: {data_technology}")

