import streamlit as st
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
import xarray as xr
from pyproj import CRS
import numpy as np
import datashader as ds
import matplotlib.colormaps
from shapely.geometry import box
import rtree

# Function to apply color map
def apply_color_map(agg):
    """Apply a color map to the aggregated data."""
    img = ds.tf.shade(
        agg,
        cmap=matplotlib.colormaps["Reds"],
        alpha=100,
        how="linear",
    )[::-1].to_pil()
    return img

# Function to read GeoTIFF to xarray
def read_geotiff_to_xarray(filepath: str) -> tuple[xr.Dataset, CRS]:
    """Read a GeoTIFF file into an xarray Dataset."""
    return xr.open_dataset(filepath).sel(band=1), get_crs(filepath)

# Function to get bounding box of a GeoTIFF tile
def get_tile_bounds(xarr_dataset: xr.Dataset) -> tuple[float, float, float, float]:
    """Get the bounding box (min_lon, min_lat, max_lon, max_lat) of a GeoTIFF tile."""
    lon = xarr_dataset['x'].values
    lat = xarr_dataset['y'].values
    return lon.min(), lat.min(), lon.max(), lat.max()

# Function to create a map with dynamic tile loading
def create_map_with_geotiff_tiles(tiles_to_overlay: list[str]) -> go.Figure:
    """Create a map with multiple GeoTIFF tiles overlaid dynamically."""
    # Create an R-tree index for the tiles
    index = rtree.index.Index()
    tile_data = []
    for idx, tile in enumerate(tiles_to_overlay):
        if tile.endswith(".tif") or tile.endswith(".tiff"):
            xarr_dataset, crs = read_geotiff_to_xarray(tile)
            min_lon, min_lat, max_lon, max_lat = get_tile_bounds(xarr_dataset)
            tile_bbox = box(min_lon, min_lat, max_lon, max_lat)
            index.insert(idx, tile_bbox.bounds)
            tile_data.append((xarr_dataset, crs))

    # Create the base map
    fig = go.Figure(go.Scattermapbox())
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=go.layout.mapbox.Center(lat=0, lon=20), zoom=8.0),
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )

    # Capture map viewport changes
    selected_points = plotly_events(fig, click_event=False, hover_event=False, relayout_event=True)

    if selected_points:
        viewport = selected_points[0].get("layout", {}).get("mapbox", {})
        center_lon, center_lat = viewport['center']['lon'], viewport['center']['lat']
        zoom = viewport['zoom']

        # Calculate the visible bounds
        delta = 180 / (2 ** zoom)
        min_lon, max_lon = center_lon - delta, center_lon + delta
        min_lat, max_lat = center_lat - delta, center_lat + delta

        # Query intersecting tiles
        viewport_bbox = box(min_lon, min_lat, max_lon, max_lat)
        intersecting_tiles = list(index.intersection(viewport_bbox.bounds))

        # Add only the visible tiles to the map
        mapbox_layers = []
        for tile_idx in intersecting_tiles:
            xarr_dataset, crs = tile_data[tile_idx]
            img = apply_color_map(xarr_dataset['band_data'])
            min_lon, min_lat, max_lon, max_lat = get_tile_bounds(xarr_dataset)
            coordinates = [
                [min_lon, min_lat],
                [min_lon, max_lat],
                [max_lon, max_lat],
                [max_lon, min_lat],
            ]
            mapbox_layers.append(
                {"sourcetype": "image", "source": img, "coordinates": coordinates}
            )

        # Update the map with visible tiles
        fig.update_layout(mapbox_layers=mapbox_layers)

    return fig

# Example usage in Streamlit
def main():
    st.title("Dynamic GeoTIFF Tile Loading")
    tiles_to_overlay = ["path/to/tile1.tif", "path/to/tile2.tif", ...]  # Add your tile paths here
    fig = create_map_with_geotiff_tiles(tiles_to_overlay)
    st.plotly_chart(fig)

if __name__ == "main":
    main()