from typing import Optional, Any
import anndata
from pathlib import Path
from abc import ABC, abstractmethod
from mlnetst.core.preprocessing.manager import PipelineStep
from mlnetst.core.preprocessing.processor import RANDOM_STATE
import decoupler as dc
import nichecompass as nc


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
    DEFAULT_EMBEDDING_KWS = {
        'net': 'collectri',
        'organism': 'mouse',
        'score': 'mlm',
        'min_n': 5,
        "verbose": False,
        'gene_orthologs_mapping_file_path': Path(__file__).parents[3] / "data" / "raw" / "human_mouse_gene_orthologs.csv",
        'random_state': RANDOM_STATE,
    }

    def __init__(self,
                 name: str,
                 output_path: Path = None,
                 force: bool = False,
                 embedding_kws: Optional[dict[str, str | float | int]] = None) -> None:
        super().__init__(name, output_path, force)
        self.embedding_kws = {**self.DEFAULT_EMBEDDING_KWS, **(embedding_kws or {})}

    def embed(self, x_hat_s: anndata.AnnData) -> anndata.AnnData:
        if self.embedding_kws.get("net") == "collectri":
            net = dc.get_collectri(
                organism = self.embedding_kws.get("organism"),
                split_complexes=False,
            )
            net["source"] = net["source"].str.lower()
            net["target"] = net["target"].str.lower()
        elif self.embedding_kws.get("net") == "omnipath":
            net = dc.omnip.
        elif self.embedding_kws.get("net") == "gp_lr":
            net = nc.utils.extract_gp_dict_from_omnipath_lr_interactions(
                species=self.embedding_kws.get("organism"),
                gene_orthologs_mapping_file_path=self.embedding_kws.get("gene_orthologs_mapping_file_path"),
            )
        elif self.embedding_kws.get("net") == "gp_tf":
            net = nc.utils.extract_gp_dict_from_collectri_tf_network(
                species=self.embedding_kws.get("organism")
            )
        elif self.embedding_kws.get("net") == "gp_es":
            net = nc.utils.extract_gp_dict_from_mebocost_es_interactions(
                species=self.embedding_kws.get("organism")
            )
        elif self.embedding_kws.get("net") == "gp_lrt":
            net = nc.utils.extract_gp_dict_from_nichenet_lrt_interactions(
                species=self.embedding_kws.get("organism"),
                gene_orthologs_mapping_file_path=self.embedding_kws.get("gene_orthologs_mapping_file_path"),
            )
        else:
            raise ValueError("Net is not supported yet. Please use collectri.")

        if self.embedding_kws.get("score") == "mlm":
            dc.run_mlm(x_hat_s,
                       net,
                       use_raw=False,
                       min_n = self.embedding_kws.get("min_n"), verbose=self.embedding_kws.get("verbose"))

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