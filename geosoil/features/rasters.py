"""Generic static-raster sampler: one function for every gridded covariate (Phase 2).

A source module (climate.py, radiometrics.py, ...) declares its layers as `Layer` records. This
module stages remote files once into data/raw/<source>/ (with a sha256 sidecar), reprojects
the sites into each layer's CRS, and samples the cell that contains each site.

Point support: the containing cell. These are coarse grids (~100 m-5 km), where interpolation
adds no information and silently smooths the nodata edges (coastlines, survey boundaries).
Sites outside a layer's extent or on nodata get NaN; that is expected (radiometrics cover only
some countries), and the model handles missing modalities.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..corpus import common
from . import shards


@dataclass(frozen=True)
class Layer:
    name: str                       # output column
    path: str                       # local path, URL to stage, or a GDAL path (/vsicurl/, NETCDF:...)
    band: int = 1
    scale: float | None = None      # None = use the file's declared scale_factor (or 1)
    offset: float | None = None     # None = use the file's declared add_offset (or 0)
    nodata: float | None = None     # overrides the file's nodata when the file doesn't declare it
    stage: bool = True              # download http(s) files first (False = read remotely via /vsicurl/)
    valid_range: tuple[float, float] | None = None   # values outside -> NaN
    filename: str | None = None     # local name when the URL path doesn't give a unique one
    subdataset: str | None = None   # NetCDF variable -> GDAL path NETCDF:<file>:<var>
    member: str | None = None       # file inside a staged zip -> GDAL path /vsizip/<zip>/<member>
    crs: str | None = None          # overrides a missing or wrong CRS in the file (any pyproj string)


@dataclass
class Source:
    name: str                       # output directory under data/processed/features/
    layers: list[Layer]
    licence: str
    citation: str = ""
    derive: object = None           # optional function(df) -> df adding derived columns
    meta: dict = field(default_factory=dict)


def _gdal_path(layer: Layer, source: str) -> str:
    p = layer.path
    if p.startswith(("http://", "https://")):
        if not layer.stage:
            p = "/vsicurl/" + p
        else:
            local = common.RAW / source / (layer.filename or Path(p.split("?")[0]).name)
            p = str(common.fetch(p, local))
    if layer.member:
        return f"/vsizip/{p}/{layer.member}"
    return f"NETCDF:{p}:{layer.subdataset}" if layer.subdataset else p


def sample_layer(gdal_path: str, layer: Layer, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Value of the cell containing each point, scaled; NaN outside the grid or on nodata."""
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import rowcol
    out = np.full(len(lon), np.nan, dtype=np.float64)
    with rasterio.open(gdal_path) as ds:
        crs = layer.crs or ds.crs
        if layer.crs or (crs and crs.to_epsg() != 4326):
            x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
        else:
            x, y = lon, lat
            # grids stored 0..360 in longitude
            if ds.bounds.left >= 0 and ds.bounds.right > 180:
                x = np.where(x < 0, x + 360, x)
        r, c = rowcol(ds.transform, x, y)
        r, c = np.asarray(r, dtype=np.int64), np.asarray(c, dtype=np.int64)
        inside = (r >= 0) & (r < ds.height) & (c >= 0) & (c < ds.width)
        if inside.any():
            nod = layer.nodata if layer.nodata is not None else ds.nodatavals[layer.band - 1]
            rr, cc = r[inside], c[inside]
            # read only the bounding window of the points, in row strips to bound memory
            vals = np.empty(len(rr), dtype=np.float64)
            order = np.argsort(rr)
            strip = 1024                     # rows per read; bounds memory on global grids
            for start in range(0, len(order), 200_000):
                idx = order[start:start + 200_000]
                r0, r1 = rr[idx].min(), rr[idx].max()
                for s0 in range(r0, r1 + 1, strip):
                    sel = idx[(rr[idx] >= s0) & (rr[idx] < s0 + strip)]
                    if not len(sel):
                        continue
                    c0, c1 = cc[sel].min(), cc[sel].max()
                    from rasterio.windows import Window
                    block = ds.read(layer.band, window=Window(c0, s0, c1 - c0 + 1,
                                                              min(strip, ds.height - s0)), masked=False)
                    vals[sel] = block[rr[sel] - s0, cc[sel] - c0]
            if nod is not None and not (isinstance(nod, float) and np.isnan(nod)):
                vals[vals == nod] = np.nan
            scale = layer.scale if layer.scale is not None else (ds.scales[layer.band - 1] or 1.0)
            offset = layer.offset if layer.offset is not None else (ds.offsets[layer.band - 1] or 0.0)
            vals = vals * scale + offset
            if layer.valid_range:
                lo, hi = layer.valid_range
                vals[(vals < lo) | (vals > hi)] = np.nan
            out[inside] = vals
    return out


def _local_file(gdal_path: str) -> Path:
    """The staged file behind a GDAL path (NETCDF:<file>:<var>, /vsizip/<zip>/<member>, or plain)."""
    if gdal_path.startswith("NETCDF:"):
        return Path(gdal_path.split(":")[1])
    if gdal_path.startswith("/vsizip/"):
        return Path(gdal_path[len("/vsizip/"):].rsplit("/", 1)[0])
    return Path(gdal_path)


def extract(source: Source, sites: pd.DataFrame, out_root: Path = C.DATA_PROC / "features") -> dict:
    """Sample every layer of `source` at every site; writes one shard with a provenance manifest."""
    import rasterio
    out = out_root / source.name
    t0 = time.time()
    df = pd.DataFrame({"site_id": sites["site_id"].to_numpy()})
    lon, lat = sites["lon"].to_numpy(float), sites["lat"].to_numpy(float)
    failed, files = [], {}
    with rasterio.Env(GDAL_HTTP_MAX_RETRY="5", GDAL_CACHEMAX=2048):
        for layer in source.layers:
            try:
                path = _gdal_path(layer, source.name)
                files[layer.name] = path
                # skip directory listings only for remote reads; local formats like EHdr (.flt + .hdr)
                # find their sidecar files through the listing
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR" if path.startswith("/vsicurl/") else "FALSE"):
                    df[layer.name] = sample_layer(path, layer, lon, lat)
            except Exception as e:
                failed.append({"block": layer.name, "error": repr(e)[:300]})
                df[layer.name] = np.nan
    if source.derive is not None:
        df = source.derive(df)
    cover = {c: round(float(df[c].notna().mean()), 4) for c in df.columns if c != "site_id"}
    shards.write(df, out, 0, 1, {
        "source": source.name, "licence": source.licence, "citation": source.citation,
        "layers": {l.name: {"path": l.path, "band": l.band, "scale": l.scale, "offset": l.offset} for l in source.layers},
        "file_sha256": {k: shards.sha256(_local_file(v)) for k, v in files.items() if _local_file(v).exists()},
        "failed_blocks": failed, "coverage": cover, "sites": int(len(df)),
        "seconds": round(time.time() - t0, 1), "point_support": "containing cell", **source.meta})
    return {"source": source.name, "sites": len(df), "layers": len(source.layers), "failed": len(failed),
            "median_coverage": float(np.median(list(cover.values()))) if cover else 0.0,
            "seconds": round(time.time() - t0, 1)}
