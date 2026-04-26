"""Geospatial helper utilities for both stocks and property projects."""
from __future__ import annotations

from typing import Callable

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    """Return a copy of ``gdf`` in the requested CRS."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame missing CRS; cannot reproject")
    if gdf.crs.to_epsg() == epsg:
        return gdf
    return gdf.to_crs(epsg)


def la_centroids_and_area(la_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute centroids and area (km^2) for each local authority polygon."""
    projected = ensure_crs(la_gdf, 27700)  # British National Grid for accuracy
    centroids = projected.centroid
    area_km2 = projected.area / 1_000_000  # m^2 → km^2
    return pd.DataFrame(
        {
            "LA_code": la_gdf["LA_code"].values,
            "centroid": centroids.to_crs(4326).values,
            "area_km2": area_km2.values,
        }
    )


def spatial_join_points(points: gpd.GeoDataFrame, la_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatial join helper (points within polygons)."""
    points = ensure_crs(points, 4326)
    la_gdf = ensure_crs(la_gdf, 4326)
    return gpd.sjoin(points, la_gdf[["LA_code", "geometry"]], how="left", predicate="within")


def nearest_distance_km(
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    filter_fn: Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame] | None = None,
) -> pd.Series:
    """Distance from each origin to the nearest destination (km)."""
    origins_proj = ensure_crs(origins, 27700)
    dest_proj = ensure_crs(destinations, 27700)
    if filter_fn is not None:
        dest_proj = filter_fn(dest_proj)
    if dest_proj.empty:
        return pd.Series([float("nan")] * len(origins), index=origins.index)
    distances = origins_proj.geometry.apply(lambda geom: dest_proj.distance(geom).min())
    return distances / 1_000


def make_point_gdf(df: pd.DataFrame, lat_col: str, lon_col: str, epsg: int = 4326) -> gpd.GeoDataFrame:
    """Construct a GeoDataFrame from latitude/longitude columns."""
    geom = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    if epsg != 4326:
        gdf = gdf.to_crs(epsg)
    return gdf
