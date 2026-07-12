"""Lazy, out-of-core data module for training METHYLVI on many ``.h5ad`` files.

This lets you train :class:`~scvi.external.METHYLVI` on collections of methylation
``.h5ad`` files (e.g. one file per brain region) *without* merging them into a
single in-memory object. Each minibatch is streamed from disk on demand.

Layout assumed here (kept deliberately simple):

- The methylated counts (``mc``) and coverage (``cov``) of each context live in
  **separate** ``.h5ad`` collections, each holding its matrix in ``.X``. So a
  two-context run uses four collections: ``mCG+mc``, ``mCG+cov``, ``mCH+mc`` and
  ``mCH+cov``.
- One methylation context (e.g. ``mCG`` or ``mCH``) per ``(mc, cov)`` pair,
  since different contexts generally have different features. Pass several
  contexts via ``collections`` to train them all **jointly**.

The lazy reads are delegated to ``adataviz.AnnDataCollection`` objects, which must
expose ``collection.adata`` (merged metadata) and
``collection[idx, :].to_memory()`` (reading the matrix from ``.X``). A plain
``anndata.AnnData`` (matrix in ``.X``) is also accepted for any ``mc``/``cov``
entry; use ``adataviz.adata.read_h5ad(path)`` to load either kind automatically.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from scipy.sparse import issparse
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler

from scvi import REGISTRY_KEYS
from scvi.external.methylvi._utils import _context_cov_key, _context_mc_key

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


def _identity(x):
    """Collate that returns the (already batched) sample unchanged.

    A module-level function (not a lambda) so it is picklable for
    ``DataLoader`` worker subprocesses.
    """
    return x


def _to_dense_float32(matrix) -> np.ndarray:
    """Return a dense ``float32`` ndarray from a (possibly sparse) matrix."""
    if issparse(matrix):
        matrix = matrix.toarray()
    return np.ascontiguousarray(matrix, dtype=np.float32)


class _MatrixSource:
    """Uniform lazy-read wrapper over a plain ``AnnData`` or an ``AnnDataCollection``.

    Both input kinds are accepted for each ``mc``/``cov`` matrix:

    - An ``adataviz.AnnDataCollection`` (out-of-core, many source files). It
      exposes ``.adata`` (merged metadata) and ``collection[rows, :].to_memory``.
    - A plain ``anndata.AnnData`` holding its matrix in ``.X``. For out-of-core
      use, open it with ``backed="r"`` (e.g. ``read_h5ad(path, backed="r")``) so
      only the requested rows are pulled from disk per minibatch; a fully
      in-memory ``AnnData`` also works. Plain ``AnnData`` objects do **not** have
      an ``.adata`` attribute, which is how the two are told apart.

    Use ``adataviz.adata.read_h5ad(path)`` to obtain either kind automatically.
    """

    def __init__(self, source, row_map=None, path=None):
        self._source = source
        # If the source came from a path, remember it so each DataLoader worker
        # subprocess can reopen its own HDF5 handle (h5py handles are not
        # fork-safe). ``_pid`` tracks the process that opened ``_source``.
        self._path = path
        self._pid = os.getpid()
        # AnnDataCollection exposes a merged `.adata`; plain AnnData does not.
        self._is_collection = hasattr(source, "adata")
        # Optional map from data-module row position -> this source's real row.
        # Used to restrict to a subset of cells and to align sources whose cells
        # are stored in a different order (aligned by name upstream). ``None``
        # means an identity mapping (the source already matches the wanted order).
        self._row_map = None if row_map is None else np.asarray(row_map, dtype=int)

    @property
    def adata(self):
        """The metadata ``AnnData`` (merged for a collection, itself for AnnData)."""
        return self._source.adata if self._is_collection else self._source

    def _live_source(self):
        """Return a source valid in the current process (reopen paths per worker)."""
        if self._path is not None and self._pid != os.getpid():
            import anndata

            self._source = anndata.read_h5ad(self._path, backed="r")
            self._is_collection = hasattr(self._source, "adata")
            self._pid = os.getpid()
        return self._source

    def read(self, rows, thread: int):
        """Return an in-memory ``AnnData`` for the given data-module ``rows``."""
        rows = np.asarray(rows, dtype=int)
        if self._row_map is not None:
            rows = self._row_map[rows]
        src = self._live_source()
        if self._is_collection:
            return src[rows, :].to_memory(thread=thread)
        if getattr(src, "isbacked", False):
            # Backed HDF5 fancy-indexing requires strictly increasing row order,
            # but minibatch indices are shuffled. Read in sorted order, then
            # restore the caller's ordering. ``src`` is the original backed
            # AnnData (never a view), so this single-level index is allowed.
            order = np.argsort(rows, kind="stable")
            mem = src[rows[order], :].to_memory()
            inv = np.empty_like(order)
            inv[order] = np.arange(order.size)
            return mem[inv].to_memory()
        return src[rows, :]


class _SubsetSequentialSampler(Sampler):
    """Yield a fixed set of indices in their given order (no shuffling)."""

    def __init__(self, indices: Sequence[int]):
        self.indices = list(indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class _SubsetRandomSampler(Sampler):
    """Yield a fixed set of indices in a new random order each epoch."""

    def __init__(self, indices: Sequence[int]):
        self.indices = np.asarray(indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices[np.random.permutation(len(self.indices))].tolist())

    def __len__(self) -> int:
        return len(self.indices)


class _CollectionMethylationDataset(Dataset):
    """A ``Dataset`` that reads a minibatch of cells from the context collections.

    ``__getitem__`` receives an array of *global* cell positions (produced by a
    :class:`~torch.utils.data.BatchSampler`) and returns a METHYLVI-ready tensor
    dictionary for that minibatch, with one ``mc``/``cov`` pair per context.
    """

    def __init__(self, datamodule: MethylVIDataModule):
        self._dm = datamodule

    def __len__(self) -> int:
        return self._dm.n_obs

    def __getitem__(self, indices) -> dict[str, torch.Tensor]:
        dm = self._dm
        idx = np.asarray(indices, dtype=int)

        tensors: dict[str, torch.Tensor] = {}
        for context in dm.contexts:
            mc = _to_dense_float32(dm.mc_collections[context].read(idx, dm.thread).X)
            cov = _to_dense_float32(dm.cov_collections[context].read(idx, dm.thread).X)
            tensors[_context_mc_key(context)] = torch.from_numpy(mc)
            tensors[_context_cov_key(context)] = torch.from_numpy(cov)

        tensors[REGISTRY_KEYS.BATCH_KEY] = torch.from_numpy(dm._batch_codes[idx]).reshape(-1, 1)

        if dm._cont_covs is not None:
            tensors[REGISTRY_KEYS.CONT_COVS_KEY] = torch.from_numpy(dm._cont_covs[idx])
        if dm._cat_codes is not None:
            tensors[REGISTRY_KEYS.CAT_COVS_KEY] = torch.from_numpy(dm._cat_codes[idx])

        return tensors


class MethylVIDataModule(LightningDataModule):
    """Out-of-core :class:`~lightning.pytorch.core.LightningDataModule` for METHYLVI.

    Wraps one ``adataviz.AnnDataCollection`` per ``(context, matrix)`` pair so
    that :class:`~scvi.external.METHYLVI` can be trained on millions of cells
    spread across many ``.h5ad`` files without loading them all into memory. Each
    context is described by **two** collections: one holding the methylated
    counts (``mc``) in ``.X`` and one holding the total coverage (``cov``) in
    ``.X``. Passing several contexts trains them jointly.

    Sources do **not** need to share a cell order: every mc/cov source and the
    metadata source is aligned **by cell name** to a common order (an explicit
    ``obs_names`` or, by default, the metadata source's cells). Sources may also
    be supersets — only the requested cells are used. Per cell metadata
    (``batch_key`` and covariates) is read from ``metadata_collection`` (default:
    the first context's ``mc`` source). Contexts may have different features.

    Parameters
    ----------
    collections
        Nested dict ``{context_name: {"mc": mc_source, "cov": cov_source}}``
        (e.g. ``{"mCG": {"mc": cg_mc, "cov": cg_cov}, "mCH": {"mc": ch_mc, "cov":
        ch_cov}}``). Each source may be a path to a ``.h5ad`` file, a plain
        ``anndata.AnnData`` (in-memory or ``backed="r"``), or an
        ``adataviz.AnnDataCollection``; in all cases the matrix is read from
        ``.X``. Paths and ``backed="r"`` AnnData stay out-of-core (only minibatch
        rows are read per step).
    mc_layer
        Name of the layer used to hold methylated-cytosine counts in the tiny
        skeleton ``MuData`` built for model construction (registry key only; the
        source collections store their matrix in ``.X``).
    cov_layer
        Name of the layer used to hold total coverage counts in the tiny skeleton
        ``MuData`` built for model construction (registry key only; the source
        collections store their matrix in ``.X``).
    metadata_collection
        Source whose ``obs`` provides ``batch_key`` and covariate columns (a
        path, AnnData or AnnDataCollection). Defaults to the first context's
        ``mc`` source.
    obs_names
        Explicit list of cell names giving the cells to use and their order. All
        sources are aligned to it by name (subsetting supersets and fixing
        differing per-file orders). If ``None``, the metadata source's cells are
        used as-is.
    batch_key
        Column in the metadata ``obs`` used as the batch covariate. If ``None``, a
        single batch is used.
    categorical_covariate_keys
        ``obs`` columns treated as (nuisance) categorical covariates.
    continuous_covariate_keys
        ``obs`` columns treated as (nuisance) continuous covariates.
    batch_size
        Minibatch size.
    train_size
        Fraction of cells used for training.
    validation_size
        Explicit validation fraction. If ``None``, uses ``1 - train_size``.
    shuffle_set_split
        Whether to shuffle before splitting into train/validation.
    n_skeleton_cells
        Number of cells materialized into the in-memory *skeleton* MuData used to
        build the model/module. The skeleton is tiny and only used for
        construction; training/inference read from disk. Its ``obs`` columns carry
        the *full* category lists so cardinalities are correct.
    thread
        Number of concurrent source-file reads per minibatch.
    num_workers
        Number of ``DataLoader`` worker subprocesses used to prefetch minibatches
        from disk (``0`` = read in the main process). Values > 0 overlap disk I/O
        with training compute and typically speed up out-of-core runs on large
        files. Each worker reopens path-based backed sources with its own HDF5
        handle; if you pass already-opened ``backed="r"`` AnnData (rather than
        paths) use ``0`` to stay safe.
    seed
        Random seed for the train/validation split.

    Examples
    --------
    >>> from adataviz.adata import AnnDataCollection
    >>> from scvi.external import METHYLVI
    >>> from scvi.external.methylvi import MethylVIDataModule
    >>>
    >>> cg_mc = AnnDataCollection.from_files("/data/*.mCG.mc.h5ad")  # matrix in .X
    >>> cg_cov = AnnDataCollection.from_files("/data/*.mCG.cov.h5ad")  # matrix in .X
    >>> ch_mc = AnnDataCollection.from_files("/data/*.mCH.mc.h5ad")  # matrix in .X
    >>> ch_cov = AnnDataCollection.from_files("/data/*.mCH.cov.h5ad")  # matrix in .X
    >>> dm = MethylVIDataModule(
    ...     collections={
    ...         "mCG": {"mc": cg_mc, "cov": cg_cov},
    ...         "mCH": {"mc": ch_mc, "cov": ch_cov},
    ...     },
    ...     batch_key="region",
    ... )
    >>> dm.register_manager(METHYLVI)
    >>> model = METHYLVI(dm.mudata)              # mCG and mCH trained jointly
    >>> model.train(max_epochs=50, datamodule=dm)
    >>> latent = model.get_latent_representation(dataloader=dm.inference_dataloader())
    """

    def __init__(
        self,
        collections: dict,
        mc_layer: str = "mc",
        cov_layer: str = "cov",
        metadata_collection=None,
        obs_names=None,
        batch_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        batch_size: int = 128,
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        n_skeleton_cells: int = 128,
        thread: int = 8,
        num_workers: int = 0,
        seed: int = 0,
    ):
        super().__init__()
        import anndata
        from mudata import MuData

        if not collections:
            raise ValueError("`collections` must contain at least one context.")

        def _as_source(x):
            # Accept a path, a plain AnnData (in-memory or backed), or an
            # AnnDataCollection. Paths are opened backed so only minibatch rows
            # are read from disk; the path is kept so DataLoader workers can
            # reopen their own handle. Returns ``(source, path_or_None)``.
            if isinstance(x, str):
                return anndata.read_h5ad(x, backed="r"), x
            return x, None

        def _obs_names_of(src):
            ad = src.adata if hasattr(src, "adata") else src
            return ad.obs_names.astype(str)

        self.contexts = list(collections.keys())
        raw_mc: dict = {}
        raw_cov: dict = {}
        raw_mc_path: dict = {}
        raw_cov_path: dict = {}
        for context, pair in collections.items():
            if not isinstance(pair, dict) or "mc" not in pair or "cov" not in pair:
                raise ValueError(
                    f"`collections['{context}']` must be a dict with 'mc' and 'cov' "
                    "AnnData/AnnDataCollection entries (or .h5ad paths)."
                )
            raw_mc[context], raw_mc_path[context] = _as_source(pair["mc"])
            raw_cov[context], raw_cov_path[context] = _as_source(pair["cov"])
        self.mc_layer = mc_layer
        self.cov_layer = cov_layer

        # source providing per-cell metadata (batch_key + covariate columns)
        if metadata_collection is not None:
            raw_meta, _ = _as_source(metadata_collection)
        else:
            raw_meta = raw_mc[self.contexts[0]]

        # Canonical cell order: an explicit `obs_names` (subset/reorder) or the
        # metadata source's cells. Every mc/cov/metadata source is aligned to it
        # BY NAME, so the sources may store cells in different orders or be
        # supersets (only these cells are used).
        if obs_names is not None:
            self.obs_names = [str(c) for c in obs_names]
        else:
            self.obs_names = list(_obs_names_of(raw_meta))
        self.n_obs = len(self.obs_names)

        def _row_map(src):
            names = _obs_names_of(src)
            pos = pd.Series(np.arange(len(names)), index=names)
            mapped = pos.reindex(self.obs_names)
            if mapped.isna().any():
                missing = mapped.index[mapped.isna()][:5].tolist()
                raise ValueError(
                    f"{int(mapped.isna().sum())} requested cell(s) are absent from a "
                    f"collection (e.g. {missing}). Every mc/cov source and the metadata "
                    "collection must contain all cells in `obs_names`."
                )
            arr = mapped.to_numpy(dtype=np.int64)
            # identity mapping -> no remap needed (keeps the fast path)
            return None if np.array_equal(arr, np.arange(len(arr))) else arr

        self.mc_collections = {
            c: _MatrixSource(raw_mc[c], _row_map(raw_mc[c]), path=raw_mc_path[c])
            for c in self.contexts
        }
        self.cov_collections = {
            c: _MatrixSource(raw_cov[c], _row_map(raw_cov[c]), path=raw_cov_path[c])
            for c in self.contexts
        }
        self.collection = _MatrixSource(raw_meta, _row_map(raw_meta))

        self.batch_key = batch_key
        self.categorical_covariate_keys = list(categorical_covariate_keys or [])
        self.continuous_covariate_keys = list(continuous_covariate_keys or [])
        self.thread = thread
        self.seed = seed
        self._num_workers = int(num_workers)

        self._batch_size = batch_size
        self._train_size = train_size
        self._validation_size = validation_size
        self._shuffle_set_split = shuffle_set_split

        # per-cell metadata aligned to the canonical cell order
        meta_src_ad = raw_meta.adata if hasattr(raw_meta, "adata") else raw_meta
        obs = meta_src_ad.obs.copy()
        obs.index = obs.index.astype(str)
        obs = obs.reindex(self.obs_names)
        self._obs = obs
        self.num_features_per_context = [
            int(self.mc_collections[ctx].adata.n_vars) for ctx in self.contexts
        ]
        self.n_vars = int(np.sum(self.num_features_per_context))

        # --- batch encoding (shared, deterministic ordering) ---
        if batch_key is not None:
            self.batch_mapping = np.sort(obs[batch_key].astype(str).unique())
            code_map = {c: i for i, c in enumerate(self.batch_mapping)}
            self._batch_codes = obs[batch_key].astype(str).map(code_map).to_numpy(dtype=np.int64)
        else:
            self.batch_mapping = np.array(["0"])
            self._batch_codes = np.zeros(self.n_obs, dtype=np.int64)
        self.n_batch = int(len(self.batch_mapping))

        # --- categorical covariate encoding ---
        self._cat_mappings: dict[str, np.ndarray] = {}
        if self.categorical_covariate_keys:
            cat_cols = []
            for key in self.categorical_covariate_keys:
                mapping = np.sort(obs[key].astype(str).unique())
                self._cat_mappings[key] = mapping
                code_map = {c: i for i, c in enumerate(mapping)}
                cat_cols.append(obs[key].astype(str).map(code_map).to_numpy(dtype=np.int64))
            self._cat_codes = np.stack(cat_cols, axis=1)
            self.n_cats_per_cov = tuple(len(m) for m in self._cat_mappings.values())
        else:
            self._cat_codes = None
            self.n_cats_per_cov = None

        # --- continuous covariates ---
        if self.continuous_covariate_keys:
            self._cont_covs = obs[self.continuous_covariate_keys].to_numpy(dtype=np.float32)
            self.n_continuous_cov = len(self.continuous_covariate_keys)
        else:
            self._cont_covs = None
            self.n_continuous_cov = 0

        self.n_labels = 1

        # --- build the tiny in-memory skeleton MuData for model construction ---
        self.mudata = self._build_skeleton(anndata, MuData, n_skeleton_cells)

        self._dataset = _CollectionMethylationDataset(self)
        self.train_idx: np.ndarray | None = None
        self.val_idx: np.ndarray | None = None
        self.test_idx: np.ndarray | None = None

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _build_skeleton(self, anndata, MuData, n_skeleton_cells):
        """Build a tiny in-memory MuData used only to construct the model/module."""
        k = int(min(n_skeleton_cells, self.n_obs))
        rows = np.arange(k)

        # obs (with batch/covariate columns) comes from the metadata, already
        # aligned to the canonical cell order.
        meta_obs = self._obs.iloc[rows].copy()
        meta_obs.index = meta_obs.index.map(str)
        if self.batch_key is not None:
            meta_obs[self.batch_key] = pd.Categorical(
                meta_obs[self.batch_key].astype(str), categories=self.batch_mapping
            )
        for key, mapping in self._cat_mappings.items():
            meta_obs[key] = pd.Categorical(meta_obs[key].astype(str), categories=mapping)

        mods = {}
        for context in self.contexts:
            mc_ad = self.mc_collections[context].read(rows, self.thread)
            cov_ad = self.cov_collections[context].read(rows, self.thread)
            ann = anndata.AnnData(
                X=mc_ad.X.copy(),
                obs=meta_obs.copy(),
                var=mc_ad.var.copy(),
            )
            ann.layers[self.mc_layer] = mc_ad.X.copy()
            ann.layers[self.cov_layer] = cov_ad.X.copy()
            mods[context] = ann

        return MuData(mods)

    # ------------------------------------------------------------------
    # convenience: register the AnnDataManager on the model class
    # ------------------------------------------------------------------
    def register_manager(self, model_cls, **setup_kwargs) -> None:
        """Run ``model_cls.setup_mudata`` on the skeleton MuData.

        Parameters
        ----------
        model_cls
            The :class:`~scvi.external.METHYLVI` class (passed in to avoid a
            circular import).
        setup_kwargs
            Extra keyword arguments forwarded to ``setup_mudata``.
        """
        mod0 = self.contexts[0]
        modalities = {"batch_key": mod0}
        if self.categorical_covariate_keys:
            modalities["categorical_covariate_keys"] = mod0
        if self.continuous_covariate_keys:
            modalities["continuous_covariate_keys"] = mod0

        model_cls.setup_mudata(
            self.mudata,
            mc_layer=self.mc_layer,
            cov_layer=self.cov_layer,
            methylation_contexts=self.contexts,
            batch_key=self.batch_key,
            categorical_covariate_keys=self.categorical_covariate_keys or None,
            continuous_covariate_keys=self.continuous_covariate_keys or None,
            modalities=modalities,
            **setup_kwargs,
        )

    # ------------------------------------------------------------------
    # hooks used by scvi's train() / TrainRunner
    # ------------------------------------------------------------------
    def set_batch_size(self, batch_size: int | None) -> None:
        """Update the minibatch size (called by ``model.train``)."""
        if batch_size is not None:
            self._batch_size = batch_size

    def set_split(
        self,
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int | None = None,
    ) -> None:
        """Update the train/validation split (called by ``model.train``)."""
        if train_size is not None:
            self._train_size = train_size
        self._validation_size = validation_size
        self._shuffle_set_split = shuffle_set_split
        if batch_size is not None:
            self._batch_size = batch_size
        self.setup()

    def setup(self, stage: str | None = None) -> None:
        """Compute train/validation/test index splits."""
        indices = np.arange(self.n_obs)
        if self._shuffle_set_split:
            indices = np.random.RandomState(seed=self.seed).permutation(indices)

        n_train = int(np.ceil(self._train_size * self.n_obs))
        if self._validation_size is None:
            n_val = self.n_obs - n_train
        else:
            n_val = int(np.floor(self._validation_size * self.n_obs))

        self.val_idx = indices[:n_val]
        self.train_idx = indices[n_val : (n_val + n_train)]
        self.test_idx = indices[(n_val + n_train) :]

    @property
    def n_train(self) -> int:
        if self.train_idx is None:
            self.setup()
        return len(self.train_idx)

    @property
    def n_val(self) -> int:
        if self.val_idx is None:
            self.setup()
        return len(self.val_idx)

    def _make_loader(self, indices, shuffle: bool, drop_last: bool) -> DataLoader:
        sampler = _SubsetRandomSampler(indices) if shuffle else _SubsetSequentialSampler(indices)
        batch_sampler = BatchSampler(sampler, batch_size=self._batch_size, drop_last=drop_last)
        kwargs = {}
        if self._num_workers > 0:
            # Overlap disk reads (each backed source reopens per worker) with the
            # main-process compute. persistent_workers avoids re-spawning/reopening
            # every epoch.
            kwargs = {"num_workers": self._num_workers, "persistent_workers": True}
        return DataLoader(
            self._dataset,
            sampler=batch_sampler,
            batch_size=None,
            collate_fn=_identity,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training data loader."""
        if self.train_idx is None:
            self.setup()
        return self._make_loader(self.train_idx, shuffle=True, drop_last=False)

    def val_dataloader(self) -> DataLoader | None:
        """Return the validation data loader (or ``None`` if empty)."""
        if self.val_idx is None:
            self.setup()
        if len(self.val_idx) == 0:
            return None
        return self._make_loader(self.val_idx, shuffle=False, drop_last=False)

    def inference_dataloader(self, indices: Sequence[int] | None = None) -> DataLoader:
        """Return a sequential loader over all cells (or ``indices``) for inference.

        Pass the result to ``model.get_latent_representation(dataloader=...)`` or
        ``model.get_elbo(dataloader=...)``. Outputs are in the same order as
        ``indices`` (default: ``0 .. n_obs - 1``).
        """
        if indices is None:
            indices = np.arange(self.n_obs)
        return self._make_loader(indices, shuffle=False, drop_last=False)
