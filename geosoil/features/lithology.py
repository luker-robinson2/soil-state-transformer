"""Parent material for every labelled site: lithology class from geological maps (Phase 2).

Parent material sets the starting texture and base status of a soil, so it is among the top texture
and pH covariates in digital soil mapping. Two maps, joined by point-in-polygon:

  global  GLiM (Hartmann & Moosdorf 2012, doi:10.1029/2012GC004370), 16 first-level classes in field
          `xx`. Polygons distributed by Universität Hamburg as an Esri FileGDB (~1.1 GB).
  US      USGS SGMC, State Geologic Map Compilation (Horton et al. 2017), 1:1M, field
          GENERALIZED_LITH (33 classes), public domain.

Columns are one-hot so tree models and networks read them without an arbitrary code order:
  litho_glim_<xx>          for each GLiM class (NaN outside polygons)
  litho_sgmc_<class>       for each SGMC class (NaN outside the conterminous US)

    python -m geosoil.features.static --source lithology
"""
from __future__ import annotations

import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..corpus import common
from . import shards

GLIM_URL = "https://www.dropbox.com/s/9vuowtebp9f1iud/LiMW_GIS%202015.gdb.zip?dl=1"
SGMC_URL = ("https://www.sciencebase.gov/catalog/file/get/5888bf4fe4b05ccb964bab9d?f=__disk__24%2Ff6%2Fe1%2F"
            "24f6e139c181fd9fe43df2aaf7f50b1c5b3b6297")
GLIM_CLASSES = {"su": "unconsolidated sediments", "ss": "siliciclastic sedimentary", "sm": "mixed sedimentary",
                "py": "pyroclastics", "sc": "carbonate sedimentary", "ev": "evaporites", "mt": "metamorphics",
                "pa": "acid plutonics", "pi": "intermediate plutonics", "pb": "basic plutonics",
                "va": "acid volcanics", "vi": "intermediate volcanics", "vb": "basic volcanics",
                "ig": "ice and glaciers", "wb": "water bodies", "nd": "no data"}
LICENCE = ("GLiM: Hartmann & Moosdorf 2012 (CC-BY 3.0 for the PANGAEA release; no licence stated on the GDB); "
           "SGMC: USGS public domain")


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _unzip_gdb(zpath: Path) -> Path:
    out = zpath.with_suffix("")
    if not out.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    gdbs = sorted(out.rglob("*.gdb"))
    if not gdbs:
        raise FileNotFoundError(f"no .gdb inside {zpath}")
    return gdbs[0]


def point_in_polygon(sites: pd.DataFrame, polys, field: str) -> pd.Series:
    """Value of `field` from the polygon containing each site (first match), NaN outside."""
    import geopandas as gpd
    pts = gpd.GeoDataFrame({"site_id": sites["site_id"].to_numpy()},
                           geometry=gpd.points_from_xy(sites["lon"], sites["lat"]), crs="EPSG:4326")
    pts = pts.to_crs(polys.crs)
    j = gpd.sjoin(pts, polys[[field, "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    return pd.Series(j[field].to_numpy(), index=sites.index)


def one_hot(values: pd.Series, prefix: str, classes=None) -> pd.DataFrame:
    """One column per class; rows with no polygon stay NaN in every column."""
    v = values.map(lambda x: slug(x) if isinstance(x, str) and x.strip() else np.nan)
    classes = classes or sorted(v.dropna().unique())
    out = pd.DataFrame({f"{prefix}{c}": (v == c).astype(float) for c in classes}, index=values.index)
    out[v.isna()] = np.nan
    return out


def extract(sites: pd.DataFrame, out_root: Path = C.DATA_PROC / "features") -> dict:
    import pyogrio
    t0, failed, parts = time.time(), [], [pd.DataFrame({"site_id": sites["site_id"].to_numpy()}, index=sites.index)]
    for name, url, field, layer, prefix, classes in (
            ("glim", GLIM_URL, "xx", None, "litho_glim_", list(GLIM_CLASSES)),
            ("sgmc", SGMC_URL, "GENERALIZED_LITH", "SGMC_Geology", "litho_sgmc_", None)):
        try:
            z = common.fetch(url, common.RAW / "lithology" / f"{name}.gdb.zip")
            gdb = _unzip_gdb(z)
            if layer is None:
                layers = [l for l, _ in pyogrio.list_layers(gdb)]
                layer = next(l for l in layers if field in pyogrio.read_info(gdb, layer=l)["fields"])
            polys = pyogrio.read_dataframe(gdb, layer=layer, columns=[field])
            parts.append(one_hot(point_in_polygon(sites, polys, field), prefix, classes))
        except Exception as e:
            failed.append({"block": name, "error": repr(e)[:300]})
    df = pd.concat(parts, axis=1).reset_index(drop=True)
    cover = {p: round(float(df[[c for c in df if c.startswith(p)]].notna().any(axis=1).mean()), 4)
             for p in ("litho_glim_", "litho_sgmc_") if any(c.startswith(p) for c in df)}
    shards.write(df, out_root / "lithology", 0, 1, {"source": "lithology", "licence": LICENCE, "coverage": cover,
                                                    "failed_blocks": failed, "sites": int(len(df)),
                                                    "seconds": round(time.time() - t0, 1)})
    return {"source": "lithology", "sites": len(df), "columns": len(df.columns) - 1, "failed": len(failed),
            "coverage": cover, "seconds": round(time.time() - t0, 1)}
