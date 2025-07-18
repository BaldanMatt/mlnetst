import marimo

__generated_with = "0.13.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import os
    import anndata
    import matplotlib.pyplot as plt
    import seaborn as sns
    return Path, anndata, os, plt, sns


@app.cell
def _(Path, os):
    MOUNT_DATASETS_PATH = Path("/media/bio/Elements/Content/SPATIALDATA")
    print(os.listdir(MOUNT_DATASETS_PATH))
    return (MOUNT_DATASETS_PATH,)


@app.cell
def _(MOUNT_DATASETS_PATH, anndata):
    snrna_data = anndata.read_h5ad(MOUNT_DATASETS_PATH / "MOp" / "snrna" / "counts100k_norm.h5ad")
    return (snrna_data,)


@app.cell
def _(snrna_data):
    snrna_data.raw = None
    print(snrna_data)
    return


@app.cell
def _(plt, snrna_data, sns):
    fig, axs = plt.subplots(ncols = 1, nrows=2, figsize=(10,10))
    sns.histplot(data = snrna_data.obs, x="Allen.subclass_label", hue="Allen.subclass_label", ax=axs[0], legend = False)
    sns.histplot(data = snrna_data.obs, x="Allen.class_label", hue="Allen.class_label", ax=axs[1], legend=False)
    return


@app.cell
def _(MOUNT_DATASETS_PATH, snrna_data):
    print(type(snrna_data.X))
    snrna_data.write_h5ad(MOUNT_DATASETS_PATH / "MOp" / "snrna" / "counts100k_norm_wo_raw.h5ad")
    return


@app.cell
def _(Path):
    import spatialdata
    import sys
    sys.path.append(str(Path(__file__).parents[1]))
    from mlnetst.core.preprocessing.manager import Builder
    return (Builder,)


@app.cell
def _(Builder, MOUNT_DATASETS_PATH):
    builder = Builder()
    file_path = MOUNT_DATASETS_PATH / "MOp" / "spatial" / "counts.zarr"
    loader = builder.produce_loader(
        name="loader",
        data_technology = "merscope",
        file_path = file_path
    )
    pipeline = builder.pipeline
    pipeline.run()
    return (loader,)


@app.cell
def _(loader):
    spatial_data = loader.outputs["data"]
    print(spatial_data.tables["table"].obs["slice_id"].astype("category").cat.categories)
    return (spatial_data,)


@app.cell
def _(spatial_data):
    slice40 = spatial_data.tables["table"][spatial_data.tables["table"].obs["slice_id"].isin(["mouse1_slice40"]),:]
    print(slice40)
    slice102 = spatial_data.tables["table"][spatial_data.tables["table"].obs["slice_id"].isin(["mouse1_slice102"]),:]
    print(slice102)
    slice153 = spatial_data.tables["table"][spatial_data.tables["table"].obs["slice_id"].isin(["mouse1_slice153"]),:]
    print(slice153)
    slice200 = spatial_data.tables["table"][spatial_data.tables["table"].obs["slice_id"].isin(["mouse1_slice200"]),:]
    print(slice200)

    return slice102, slice153, slice200, slice40


@app.cell
def _(MOUNT_DATASETS_PATH, slice102, slice153, slice200, slice40):
    slice40.write_h5ad(MOUNT_DATASETS_PATH / "MOp" / "spatial" / "slice40_norm_wo_raw.h5ad")
    slice102.write_h5ad(MOUNT_DATASETS_PATH / "MOp" / "spatial" / "slice102_norm_wo_raw.h5ad")
    slice153.write_h5ad(MOUNT_DATASETS_PATH / "MOp" / "spatial" / "slice153_norm_wo_raw.h5ad")
    slice200.write_h5ad(MOUNT_DATASETS_PATH / "MOp" / "spatial" / "slice200_norm_wo_raw.h5ad")
    return


if __name__ == "__main__":
    app.run()
