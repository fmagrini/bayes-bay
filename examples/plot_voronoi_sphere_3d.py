"""Plot ``Voronoi2DSphere`` with and without optional geographic context.

The first figure uses only BayesBay's required dependencies. When Cartopy is
installed, a second figure adds Natural Earth coastlines. Cartopy supplies the
geographic data; Matplotlib performs the three-dimensional rendering.
"""

import matplotlib.pyplot as plt
import numpy as np

from bayesbay.discretization import Voronoi2DSphere

try:  # Cartopy is optional and is not a BayesBay dependency.
    import cartopy.io.shapereader as shapereader
except ImportError:
    shapereader = None


def create_example_tessellation(seed=7, n_sites=60):
    """Create uniformly distributed sites and synthetic cell values."""
    rng = np.random.default_rng(seed)
    sites = np.column_stack(
        (
            rng.uniform(-180, 180, n_sites),
            np.degrees(np.arcsin(rng.uniform(-1, 1, n_sites))),
        )
    )
    values = 3 + 0.3 * np.sin(np.radians(2 * sites[:, 0])) * np.cos(
        np.radians(sites[:, 1])
    )
    return sites, values


def create_tessellation_figure(
    sites,
    values,
    title,
    boundary_linewidth=0.8,
    show_sites=False,
):
    """Create one Matplotlib 3-D tessellation figure."""
    figure = plt.figure(figsize=(8, 7))
    axes = figure.add_subplot(projection="3d")
    site_style = (
        {"color": "white", "edgecolor": "black", "s": 12} if show_sites else None
    )
    Voronoi2DSphere.plot_tessellation_3d(
        sites,
        values,
        ax=axes,
        cmap="viridis",
        surface_spacing_deg=1,
        linewidth=boundary_linewidth,
        voronoi_sites_kwargs=site_style,
        zorder=1,
    )
    axes.set_title(title)
    # On the unit sphere, Matplotlib's azimuth and elevation correspond to
    # longitude and latitude. This initial view is centred on Italy.
    axes.view_init(elev=42.5, azim=12.5)
    return figure, axes


def load_natural_earth_coastlines(radius=1.015):
    """Load Natural Earth coastlines and convert them to Cartesian paths."""
    coastline_file = shapereader.natural_earth(
        resolution="110m", category="physical", name="coastline"
    )
    coastline_paths = []
    for geometry in shapereader.Reader(coastline_file).geometries():
        line_strings = geometry.geoms if hasattr(geometry, "geoms") else [geometry]
        for line_string in line_strings:
            longitude_latitude = np.asarray(line_string.coords)
            path = Voronoi2DSphere.lonlat_to_xyz(longitude_latitude) * radius
            coastline_paths.append(path)
    return coastline_paths


def camera_direction(axes):
    """Return the unit vector pointing from the globe towards the camera."""
    elevation = np.radians(axes.elev)
    azimuth = np.radians(axes.azim)
    return np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )


def front_facing_sections(path, view_direction):
    """Split a Cartesian path into sections on the camera-facing hemisphere."""
    is_front_facing = path @ view_direction > 0
    front_indices = np.flatnonzero(is_front_facing)
    if not front_indices.size:
        return []
    section_breaks = np.flatnonzero(np.diff(front_indices) > 1) + 1
    return [
        path[index_section]
        for index_section in np.split(front_indices, section_breaks)
        if len(index_section) >= 2
    ]


def add_coastline_overlay(axes, coastline_paths):
    """Add coastlines and keep them aligned when the globe is rotated.

    Matplotlib does not reliably depth-sort 3-D lines against a surface. The
    overlay therefore uses an explicit z-order and omits paths on the far side
    of the globe. The front-facing sections are recalculated after rotation.
    """
    axes.computed_zorder = False
    coastline_artists = []

    def update_coastline_lines():
        while coastline_artists:
            coastline_artists.pop().remove()
        view_direction = camera_direction(axes)
        for path in coastline_paths:
            for section in front_facing_sections(path, view_direction):
                (line,) = axes.plot(
                    section[:, 0],
                    section[:, 1],
                    section[:, 2],
                    color="white",
                    linewidth=0.9,
                    alpha=0.95,
                    zorder=10,
                )
                coastline_artists.append(line)

    def handle_rotation(event):
        if event.inaxes is axes:
            update_coastline_lines()
            axes.figure.canvas.draw_idle()

    update_coastline_lines()
    axes.figure.canvas.mpl_connect("button_release_event", handle_rotation)


def main():
    sites, values = create_example_tessellation()

    create_tessellation_figure(
        sites,
        values,
        title="Voronoi2DSphere — Matplotlib",
    )

    if shapereader is None:
        print("Cartopy is not installed; skipping the coastline figure.")
    else:
        try:
            coastline_paths = load_natural_earth_coastlines()
        except Exception as exc:
            # Natural Earth data may need to be downloaded on first use. The
            # Matplotlib-only figure remains usable in offline environments.
            print(f"Could not load optional Cartopy coastlines: {exc}")
        else:
            _, coastline_axes = create_tessellation_figure(
                sites,
                values,
                title="Voronoi2DSphere — Cartopy Natural Earth coastlines",
                boundary_linewidth=0,
                show_sites=False,
            )
            add_coastline_overlay(coastline_axes, coastline_paths)

    plt.show()


if __name__ == "__main__":
    main()
