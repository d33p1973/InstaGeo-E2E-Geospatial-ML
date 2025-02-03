import glob
import json
import os
from pathlib import Path
from functools import lru_cache

import streamlit as st
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.cm
import plotly.graph_objects as go
import rasterio
import xarray as xr
from pyproj import CRS, Transformer

epsg3857_to_epsg4326 = Transformer.from_crs(3857, 4326, always_xy=True)

# Cache for storing loaded tiles
@lru_cache(maxsize=32)
def read_geotiff_to_xarray_cached(filepath: str) -> tuple[xr.Dataset, CRS]:
    """Read a GeoTIFF file into an xarray Dataset with caching.

    Args:
        filepath (str): Path to the GeoTIFF file.

    Returns:
        xr.Dataset: The loaded xarray dataset.
    """
    return xr.open_dataset(filepath).sel(band=1), get_crs(filepath)

def get_crs(filepath: str) -> CRS:
    """Retrieves the CRS of a GeoTiff data.

    Args:
        filepath: Path to a GeoTiff file.

    Returns:
        CRS of data stored in `filepath`
    """
    src = rasterio.open(filepath)
    return src.crs

def add_raster_to_plotly_figure(
    xarr_dataset: xr.Dataset,
    from_crs: CRS,
    column_name: str = "band_data",
    scale: float = 1.0,
) -> tuple:
    """Add a raster plot on a Plotly graph object figure.

    Args:
        xarr_dataset (xr.Dataset): xarray dataset containing the raster data.
        from_crs (CRS): Coordinate Reference System of data stored in xarr_dataset.
        column_name (str): Name of the column in `xarr_dataset` to be plotted.
        scale (float): Scale factor for adjusting the plot resolution.

    Returns:
        tuple: (PIL.Image, list) The image and its coordinates.
    """
    xarr_dataset = xarr_dataset.rio.write_crs(from_crs).rio.reproject("EPSG:3857")
    xarr_dataset = xarr_dataset.where(xarr_dataset <= 1, 0)
    numpy_data = xarr_dataset[column_name].squeeze().to_numpy()
    plot_height, plot_width = numpy_data.shape

    canvas = ds.Canvas(
        plot_width=int(plot_width * scale), plot_height=int(plot_height * scale)
    )
    agg = canvas.raster(xarr_dataset[column_name].squeeze(), interpolate="linear")

    coords_lat_min, coords_lat_max = agg.coords["y"].values.min(), agg.coords["y"].values.max()
    coords_lon_min, coords_lon_max = agg.coords["x"].values.min(), agg.coords["x"].values.max()

    (coords_lon_min, coords_lon_max), (coords_lat_min, coords_lat_max) = epsg3857_to_epsg4326.transform(
        [coords_lon_min, coords_lon_max], [coords_lat_min, coords_lat_max]
    )

    coordinates = [
        [coords_lon_min, coords_lat_max],
        [coords_lon_max, coords_lat_max],
        [coords_lon_max, coords_lat_min],
        [coords_lon_min, coords_lat_min],
    ]

    img = tf.shade(
        agg,
        cmap=matplotlib.colormaps["Reds"],
        alpha=100,
        how="linear",
    )[::-1].to_pil()
    return img, coordinates

def load_tile_metadata(json_path: str) -> list:
    """Load tile metadata from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing tile metadata.

    Returns:
        list: List of tile metadata.
    """
    with open(json_path, "r") as f:
        return json.load(f)

def is_tile_in_viewport(tile_bounds: dict, viewport: dict) -> bool:
    """Check if a tile is within the current viewport.

    Args:
        tile_bounds (dict): Bounding box of the tile.
        viewport (dict): Current viewport of the map.

    Returns:
        bool: True if the tile is within the viewport, False otherwise.
    """
    lat_min, lat_max = viewport['latitude']['min'], viewport['latitude']['max']
    lon_min, lon_max = viewport['longitude']['min'], viewport['longitude']['max']

    tile_lat_min, tile_lat_max = tile_bounds['lat_min'], tile_bounds['lat_max']
    tile_lon_min, tile_lon_max = tile_bounds['lon_min'], tile_bounds['lon_max']

    # Check for intersection
    return not (tile_lat_max < lat_min or tile_lat_min > lat_max or
                tile_lon_max < lon_min or tile_lon_min > lon_max)

def create_map_with_geotiff_tiles(tile_metadata: list, viewport: dict, zoom: float, base_dir: str) -> go.Figure:
    """Create a map with multiple GeoTIFF tiles overlaid.

    Args:
        tile_metadata (list): List of tile metadata.
        viewport (dict): Current viewport of the map.
        zoom (float): Current zoom level of the map.
        base_dir (str): Base directory containing the tiles.

    Returns:
        go.Figure: A Plotly figure with overlaid GeoTIFF tiles.
    """
    fig = go.Figure(go.Scattermapbox())
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=go.layout.mapbox.Center(lat=0, lon=20), zoom=zoom),
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    mapbox_layers = []

    for tile in tile_metadata:
        if is_tile_in_viewport(tile['bounds'], viewport):
            tile_path = os.path.join(base_dir, tile['name'])
            xarr_dataset, crs = read_geotiff_to_xarray_cached(tile_path)
            img, coordinates = add_raster_to_plotly_figure(
                xarr_dataset, crs, "band_data", scale=1.0 #if zoom > 8 else 0.5  # Downsample when zoomed out
            )
            mapbox_layers.append(
                {"sourcetype": "image", "source": img, "coordinates": coordinates}
            )
    fig.update_layout(mapbox_layers=mapbox_layers)
    return fig

def generate_map(
    directory: str, year: int, month: int, viewport: dict, zoom: float, tile_metadata: list
) -> None:
    """Generate the plotly map.

    Args:
        directory (str): Directory containing GeoTiff files.
        year (int): Selected year.
        month (int): Selected month formatted as an integer in the range 1-12.
        viewport (dict): Current viewport of the map.
        zoom (float): Current zoom level of the map.
        tile_metadata (list): List of tile metadata.
    """
    try:
        if not directory or not Path(directory).is_dir():
            raise ValueError("Invalid directory path.")

        base_dir = os.path.join(directory, f"{year}/{month}")
        fig = create_map_with_geotiff_tiles(tile_metadata, viewport, zoom, base_dir)
        st.plotly_chart(fig, use_container_width=True)
    except (ValueError, FileNotFoundError, Exception) as e:
        st.error(f"An error occurred: {str(e)}")

def main() -> None:
    """Instageo Serve Main Entry Point."""
    st.set_page_config(layout="wide")
    st.title("InstaGeo Serve")

    st.sidebar.subheader(
        "This application enables the visualisation of GeoTIFF files on an interactive map.",
        divider="rainbow",
    )
    st.sidebar.header("Settings")

    with st.sidebar.container():
        directory = st.sidebar.text_input(
            "GeoTiff Directory:",
            help="Write the path to the directory containing your GeoTIFF files",
        )
        year = st.sidebar.number_input("Select Year", 2023, 2024)
        month = st.sidebar.number_input("Select Month", 1, 12)

    # Load tile metadata from JSON file
    tile_metadata = load_tile_metadata("tile_metadata.json")

    # Initialize viewport and zoom level in session state
    if 'viewport' not in st.session_state:
        st.session_state.viewport = {
            'latitude': {'min': -2.91785776125, 'max': -1.13465911215},
            'longitude': {'min': 29.0249263852, 'max': 30.8161348813}
        }
    if 'zoom' not in st.session_state:
        st.session_state.zoom = 8.0

    # Capture the current viewport and zoom level from the map
    if st.session_state.get('map_fig'):
        relayout_data = st.session_state.map_fig.layout.mapbox
        if relayout_data:
            st.session_state.viewport = {
                'latitude': {'min': relayout_data.center.lat - 0.1, 'max': relayout_data.center.lat + 0.1},
                'longitude': {'min': relayout_data.center.lon - 0.1, 'max': relayout_data.center.lon + 0.1}
            }
            st.session_state.zoom = relayout_data.zoom

    if st.sidebar.button("Generate Map"):
        generate_map(directory, year, month, st.session_state.viewport, st.session_state.zoom, tile_metadata)
    else:
        fig = create_map_with_geotiff_tiles(tile_metadata=[], viewport=st.session_state.viewport, zoom=st.session_state.zoom, base_dir="")
        st.session_state.map_fig = fig
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()