import numpy as np
import pytest
from mudata import MuData

from scvi.data import synthetic_iid
from scvi.external import METHYLVI


@pytest.mark.parametrize("dispersion", ["region", "region-cell"])
def test_methylvi_dispersion(dispersion: str):
    adata1 = synthetic_iid()
    adata1.layers["mc"] = adata1.X
    adata1.layers["cov"] = adata1.layers["mc"] + 10

    adata2 = synthetic_iid()
    adata2.layers["mc"] = adata2.X
    adata2.layers["cov"] = adata2.layers["mc"] + 10

    mdata = MuData({"mod1": adata1, "mod2": adata2})

    METHYLVI.setup_mudata(
        mdata,
        mc_layer="mc",
        cov_layer="cov",
        methylation_contexts=["mod1", "mod2"],
        batch_key="batch",
        modalities={"batch_key": "mod1"},
    )
    vae = METHYLVI(mdata, dispersion=dispersion)
    vae.train(1)


def test_methylvi():
    adata1 = synthetic_iid()
    adata1.layers["mc"] = adata1.X
    adata1.layers["cov"] = adata1.layers["mc"] + 10

    adata2 = synthetic_iid()
    adata2.layers["mc"] = adata2.X
    adata2.layers["cov"] = adata2.layers["mc"] + 10

    mdata = MuData({"mod1": adata1, "mod2": adata2})

    METHYLVI.setup_mudata(
        mdata,
        mc_layer="mc",
        cov_layer="cov",
        methylation_contexts=["mod1", "mod2"],
        batch_key="batch",
        modalities={"batch_key": "mod1"},
    )
    vae = METHYLVI(
        mdata,
    )
    vae.train(3)
    vae.get_elbo(indices=vae.validation_indices)
    vae.get_normalized_methylation()  # Retrieve methylation for all contexts
    vae.get_normalized_methylation(context="mod1")  # Retrieve for specific context
    with pytest.raises(ValueError):  # Should fail when invalid context selected
        vae.get_normalized_methylation(context="mod3")
    vae.get_latent_representation()
    vae.differential_methylation(groupby="mod1:labels", group1="label_1")
    vae.differential_methylation(groupby="mod1:labels", group1="label_1", two_sided=False)


def test_methylvi_covariates():
    rng = np.random.default_rng(0)

    adata1 = synthetic_iid()
    adata1.layers["mc"] = adata1.X
    adata1.layers["cov"] = adata1.layers["mc"] + 10
    adata1.obs["cont1"] = rng.normal(size=adata1.n_obs).astype("float32")
    adata1.obs["cont2"] = rng.normal(size=adata1.n_obs).astype("float32")
    adata1.obs["cat1"] = rng.integers(0, 3, size=adata1.n_obs)

    adata2 = synthetic_iid()
    adata2.layers["mc"] = adata2.X
    adata2.layers["cov"] = adata2.layers["mc"] + 10

    mdata = MuData({"mod1": adata1, "mod2": adata2})

    METHYLVI.setup_mudata(
        mdata,
        mc_layer="mc",
        cov_layer="cov",
        methylation_contexts=["mod1", "mod2"],
        batch_key="batch",
        categorical_covariate_keys=["cat1"],
        continuous_covariate_keys=["cont1", "cont2"],
        modalities={
            "batch_key": "mod1",
            "categorical_covariate_keys": "mod1",
            "continuous_covariate_keys": "mod1",
        },
    )
    assert mdata["mod1"].n_obs == mdata.n_obs
    assert METHYLVI(mdata).summary_stats["n_extra_continuous_covs"] == 2

    vae = METHYLVI(mdata)
    vae.train(3)
    vae.get_elbo(indices=vae.validation_indices)
    vae.get_normalized_methylation()
    vae.get_latent_representation()


def _write_matrix_h5ad(path, region, kind, n_features=None, n_cells=50, seed=0):
    """Write a small single-matrix methylation `.h5ad` (matrix in `.X`).

    `kind` is ``"mc"`` (methylated counts) or ``"cov"`` (coverage).
    """
    adata = synthetic_iid(batch_size=n_cells, n_batches=1)
    if n_features is not None:
        adata = adata[:, :n_features].copy()
    if kind == "cov":
        adata.X = adata.X + 10
    adata.obs["region"] = region
    adata.write_h5ad(path)


def test_methylvi_anndatacollection(save_path):
    """Train METHYLVI out-of-core over AnnDataCollections, mCG and mCH jointly."""
    adataviz_adata = pytest.importorskip("adataviz.adata")
    import os

    from scvi.external.methylvi import MethylVIDataModule

    collections = {}
    for context, n_features in [("mCG", 40), ("mCH", 25)]:
        mc_paths, cov_paths = [], []
        for i, region in enumerate(["region_A", "region_B", "region_C"]):
            mc_p = os.path.join(save_path, f"{context}_{region}.mc.h5ad")
            cov_p = os.path.join(save_path, f"{context}_{region}.cov.h5ad")
            _write_matrix_h5ad(mc_p, region, "mc", n_features=n_features, seed=i)
            _write_matrix_h5ad(cov_p, region, "cov", n_features=n_features, seed=i)
            mc_paths.append(mc_p)
            cov_paths.append(cov_p)
        collections[context] = {
            "mc": adataviz_adata.AnnDataCollection.from_files(mc_paths),
            "cov": adataviz_adata.AnnDataCollection.from_files(cov_paths),
        }

    dm = MethylVIDataModule(
        collections=collections,
        batch_key="region",
        batch_size=32,
        n_skeleton_cells=16,
    )
    assert dm.contexts == ["mCG", "mCH"]
    assert dm.n_batch == 3
    assert dm.n_obs == 150
    assert dm.num_features_per_context == [40, 25]

    dm.register_manager(METHYLVI)
    model = METHYLVI(dm.mudata)
    model.train(max_epochs=2, datamodule=dm)

    latent = model.get_latent_representation(dataloader=dm.inference_dataloader())
    assert latent.shape[0] == dm.n_obs


def test_methylvi_plain_h5ad(save_path):
    """Train METHYLVI where each mc/cov source is a plain backed `.h5ad` (matrix in .X)."""
    adataviz_adata = pytest.importorskip("adataviz.adata")
    import os

    from scvi.external.methylvi import MethylVIDataModule

    collections = {}
    for context, n_features in [("mCG", 40), ("mCH", 25)]:
        mc_p = os.path.join(save_path, f"{context}.plain.mc.h5ad")
        cov_p = os.path.join(save_path, f"{context}.plain.cov.h5ad")
        _write_matrix_h5ad(mc_p, "region_A", "mc", n_features=n_features)
        _write_matrix_h5ad(cov_p, "region_A", "cov", n_features=n_features)
        collections[context] = {
            "mc": adataviz_adata.read_h5ad(mc_p, backed="r"),
            "cov": adataviz_adata.read_h5ad(cov_p, backed="r"),
        }

    dm = MethylVIDataModule(
        collections=collections,
        batch_key="region",
        batch_size=32,
        n_skeleton_cells=16,
    )
    assert dm.contexts == ["mCG", "mCH"]
    assert dm.n_obs == 50
    assert dm.num_features_per_context == [40, 25]

    dm.register_manager(METHYLVI)
    model = METHYLVI(dm.mudata)
    model.train(max_epochs=2, datamodule=dm)

    latent = model.get_latent_representation(dataloader=dm.inference_dataloader())
    assert latent.shape[0] == dm.n_obs



