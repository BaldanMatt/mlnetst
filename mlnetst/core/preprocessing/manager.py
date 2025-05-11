from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, List, Dict

class PipelineStep(ABC):
    PROJECT_DIR = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_DIR / 'data'
    PROCESSED_DIR = DATA_DIR / 'processed'
    MEDIA_DIR = DATA_DIR / 'media'
    RESULTS_DIR = DATA_DIR / 'results'
    def __init__(self, name):
        self.name = name
        self.dependencies = []
        self.status = "pending" # pending, running, completed, failed
        self.result = None
        self._outputs: Dict[str, Any] = {}

    @property
    def outputs(self) -> Dict[str, Any]:
        return self._outputs

    def get_input_from(self,
                       dependency_name: str,
                       key: str = "default") -> Any:
        dependency = next((dep for dep in self.dependencies if dep.name == dependency_name), None)
        if not dependency:
            raise ValueError(f"Dependency '{dependency_name}' not found in pipeline step '{self.name}'.")
        if key not in dependency.outputs:
            raise ValueError(f"Key '{key}' not found in outputs of dependency '{dependency_name}'.")
        return dependency.outputs[key]

    def set_output(self, key: str, value: Any) -> None:
        self._outputs[key] = value

    def add_dependency(self, dependency):
        self.dependencies.append(dependency)

    @abstractmethod
    def run(self) -> None:
        pass

class Pipeline:
    def __init__(self) -> None:
        self.steps = []

    def add_step(self, step: PipelineStep) -> None:
        self.steps.append(step)

    def resolve_execution_order(self) -> List[PipelineStep]:
        resolved = []
        unresolved = set(self.steps)
        while unresolved:
            progress_made = False
            for step in list(unresolved):
                if all(dep in resolved for dep in step.dependencies):
                    resolved.append(step)
                    unresolved.remove(step)
                    progress_made = True
            if not progress_made:
                raise RuntimeError("Circular dependency detected in pipeline steps!")
        return resolved

    def run(self) -> None:
        ordered_steps = self.resolve_execution_order()
        total_steps = len(ordered_steps)

        print(f"Executing {self}")
        for i, step in enumerate(ordered_steps,1):
            print(f"[{i}/{total_steps}] Running step: {step.name}")
            step.status = "running"
            try:
                step.run()
                step.status = "completed"
                print(f"{step.name} completed")
            except Exception as e:
                step.status = "failed"
                print(f"Error in {step.name}: {e}")
                raise e

    def __str__(self):
        steps = self.resolve_execution_order()
        step_strings = []
        for i, step in enumerate(steps, 1):
            step_strings.append(f"Step {i}: {step}")
        return "Pipeline Steps:\n\t" + "\n\t".join(step_strings)


class Builder:
    def __init__(self):
        self._pipeline = Pipeline()

    def reset(self):
        self._pipeline = Pipeline()

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def produce_loader(self, name:str, data_technology:str, file_path: Path = None) -> PipelineStep:
        from mlnetst.core.preprocessing.loader import DataLoaderFactory
        step = DataLoaderFactory.produce_loader(name, data_technology, file_path)
        self._pipeline.add_step(step)
        return step

    def produce_preprocessor(self, name:str, data_technology: str, output_path:Path=None, **kwargs) -> PipelineStep:
        from mlnetst.core.preprocessing.processor import ProcessorFactory
        step = ProcessorFactory.produce_processor(name, data_technology, output_path, **kwargs)
        self._pipeline.add_step(step)
        return step
    def produce_integrator(self, name:str, integration_method: str, output_path: Path=None, **kwargs) -> PipelineStep:
        from mlnetst.core.preprocessing.integrator import IntegratorFactory
        step = IntegratorFactory.produce_integrator(name, integration_method, output_path, **kwargs)
        self._pipeline.add_step(step)
        return step
    def produce_embedder(self, name: str, embed_method: str, output_path: Path = None, **kwargs) -> PipelineStep:
        from mlnetst.core.preprocessing.embedder import EmbedderFactory
        step = EmbedderFactory.produce_embedder(name, embed_method, output_path, **kwargs)
        self._pipeline.add_step(step)
        return step

if __name__ == "__main__":
    builder = Builder()
    root_media_dir = Path("/media/bio/Elements/Content")
    root_project_dir = Path(__file__).parents[3]
    # root_media_dir = Path("/media/matteo/Content")
    # file_path = root_media_dir / Path("SPATIALDATA/MOp/snrna/counts100k.h5ad")
    # loader = builder.produce_loader(
    #     name="loader1",
    #     data_technology="snrna",
    #     file_path=file_path
    # )
    # file_path = root_media_dir / Path("SPATIALDATA/MOp/spatial/counts.zarr")
    # loader2 = builder.produce_loader(
    #     name="loader2",
    #     data_technology="merscope",
    #     file_path=file_path
    # )
    # output_path = root_project_dir / Path("data/processed/counts100k_processed.h5ad")
    # processor = builder.produce_preprocessor(
    #     name="processor1",
    #     data_technology="snrna",
    #     filter_method="default",
    #     norm_method="scanpy",
    #     output_path=output_path
    # )
    # output_path = root_project_dir / Path("data/processed/counts.zarr")
    # processor2 = builder.produce_preprocessor(
    #     name="processor2",
    #     data_technology="merscope",
    #     output_path=output_path
    # )
    # processor2.add_dependency(loader2)
    output_path = root_project_dir / Path("data/processed/mouse1_slice153_x_hat_s.h5ad")
    integrator = builder.produce_integrator(
        name="integrator",
        integration_method="tangram",
        output_path=output_path,
        force=False,
        integration_kws={
            'point_of_view': 'slices',
            'which_one': "mouse1_slice153",
            'subset_scale_factor': 1.5,
            'num_epochs': 100,
        }
    )
    # integrator.add_dependency(processor)
    # integrator.add_dependency(processor2)
    embedder = builder.produce_embedder(
        name="embedder",
        embed_method="decoupler",
        embedding_kws = {
            'net':'gp_lr',
            'score':'mlm',
        },
        output_path=None,
        force=True,
    )
    embedder.add_dependency(integrator)
    pipeline = builder.pipeline
    pipeline.run()
    print(embedder.outputs["embedded_data"])
    print("Pipeline execution completed.")