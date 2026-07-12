# Training METHYLVI out-of-core with `AnnDataCollection`

Use `MethylVIDataModule` to train `METHYLVI` on collections of methylation
`.h5ad` files (e.g. one file per brain region) **without merging them into
memory**. Each minibatch is streamed from disk on demand.

Assumed layout:

- The methylated counts (`mc`) and coverage (`cov`) of each context live in
  **separate** `.h5ad` collections, each holding its matrix in `.X`. So a
  two-context run uses four collections: `mCG+mc`, `mCG+cov`, `mCH+mc` and
  `mCH+cov`.
- One methylation context (e.g. `mCG` or `mCH`) per `(mc, cov)` pair, since
  contexts usually have different features. Pass several contexts to train them
  **jointly**.

All collections must describe the **same cells in the same row order**; only the
features differ between contexts.

Each `mc`/`cov` source can be either an `AnnDataCollection` (out-of-core over many
files) **or** a plain `.h5ad` loaded as an `anndata.AnnData` (matrix in `.X`). Use
`adataviz.adata.read_h5ad(path)`, which auto-detects the file type and returns the
right kind, so the same `MethylVIDataModule` works with both. For a plain `.h5ad`,
pass `backed="r"` so it stays **out-of-core** and only the minibatch rows are read
from disk each step (a fully in-memory `AnnData` also works).

## How you use it

```python
from adataviz.adata import AnnDataCollection, read_h5ad
from scvi.external import METHYLVI
from scvi.external.methylvi import MethylVIDataModule

# Out-of-core collections (many files):
cg_mc = AnnDataCollection.from_files("/data/*.mCG.mc.h5ad")    # matrix in .X
cg_cov = AnnDataCollection.from_files("/data/*.mCG.cov.h5ad")  # matrix in .X

# ...or plain single .h5ad files (auto-detected; backed keeps them out-of-core):
ch_mc = read_h5ad("/data/mCH.mc.h5ad", backed="r")    # AnnData (backed) or AnnDataCollection
ch_cov = read_h5ad("/data/mCH.cov.h5ad", backed="r")

dm = MethylVIDataModule(
    collections={
        "mCG": {"mc": cg_mc, "cov": cg_cov},
        "mCH": {"mc": ch_mc, "cov": ch_cov},
    },
    batch_key="region",          # or a technical-batch column; see note below
    batch_size=128,
)
dm.register_manager(METHYLVI)     # runs METHYLVI.setup_mudata on the skeleton
model = METHYLVI(dm.mudata)       # mCG and mCH trained jointly
model.train(max_epochs=50, datamodule=dm)          # streams cells from disk

latent = model.get_latent_representation(dataloader=dm.inference_dataloader())
```

For a single context, pass a one-entry dict, e.g.
`collections={"mCG": {"mc": cg_mc, "cov": cg_cov}}`.

Per-cell metadata (`batch_key`, covariates) is read from `metadata_collection`
(defaults to the first context's `mc` collection).

## Note on `batch_key`

If brain-region differences are biological signal you want to **keep**, do not put
`region` in `batch_key`; use a real technical-batch column instead (or
`batch_key=None`). The data module supports either.
