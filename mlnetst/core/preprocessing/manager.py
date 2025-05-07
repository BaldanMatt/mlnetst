from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, List

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

        print(f"Executing pipeline in order: {[str(step) for step in ordered_steps]}")
        for i, step in enumerate(ordered_steps,1):
            print(f"[{i}/{total_steps}] Running step: {step.name}")
            try:
                step.run()
                print(f"{step.name} completed")
            except Exception as e:
                step.status = "failed"
                print(f"Error in {step.name}: {e}")
                raise e

    def __str__(self):
        return f"The Pipeline contains the following order of steps: \n\t{'\n\t'.join([str(step) for step in self.resolve_execution_order()])}"

class Builder:
    def __init__(self):
        self._pipeline = Pipeline()

    def reset(self):
        self._pipeline = Pipeline()

    @property
    def pipeline(self) -> Pipeline:
        pipeline = self._pipeline
        return pipeline

    def produce_loader(self, name:str, data_technology:str, file_path: Path = None) -> PipelineStep:
        from mlnetst.core.preprocessing.loader import DataLoaderFactory, DataLoader
        step = DataLoaderFactory.produce_loader(name, data_technology, file_path)
        self._pipeline.add_step(step)
        return step

    def produce_preprocessor(self) -> PipelineStep:
        pass
    def produce_integrator(self) -> PipelineStep:
        pass
    def produce_embedder(self) -> PipelineStep:
        pass



if __name__ == "__main__":
    builder = Builder()
    file_path = Path("/media/matteo/Content/SPATIALDATA/MOp/snrna/counts100k.h5ad")
    loader = builder.produce_loader(
        name="loader1",
        data_technology="snrna",
        file_path=file_path
    )
    pipeline = builder.pipeline
    pipeline.run()
    print("Pipeline execution completed.")