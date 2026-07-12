"""Examples of the tessellation plotting provided by Voronoi2D and
Voronoi2DSphere, with and without a polygon delimiting the region of
interest. Figures are saved in the working directory."""
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry
import cartopy.crs as ccrs
from cmcrameri import cm as scm

from bayesbay.discretization import Voronoi2D, Voronoi2DSphere

rng = np.random.default_rng(1)


def random_sites_on_sphere(n):
    return np.column_stack(
        (rng.uniform(-180, 180, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n))))
    )


# ----------------------------------------------------------------------------
# Example 1 -- GLOBAL tessellation, plain matplotlib axes (no cartopy needed).
# Cells containing the poles and cells crossing the +/-180 meridian are drawn
# correctly.
# ----------------------------------------------------------------------------
sites = random_sites_on_sphere(80)
values = 3 + 0.3 * np.sin(np.radians(3 * sites[:, 0])) * np.sin(
    np.radians(2 * sites[:, 1])
)

fig, ax = plt.subplots(figsize=(11, 5.5))
Voronoi2DSphere.plot_tessellation(sites, values, ax=ax, cmap=scm.roma)
ax.set_title("Global tessellation, plain matplotlib axes (80 cells)")
fig.savefig("example1_global_plain", dpi=140, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# Example 2 -- the SAME global tessellation on a cartopy Robinson projection:
# just pass a GeoAxes and the transform; bayesbay never imports cartopy.
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(11, 6))
ax = plt.axes(projection=ccrs.Robinson())
Voronoi2DSphere.plot_tessellation(
    sites, values, ax=ax, cmap=scm.roma, transform=ccrs.PlateCarree()
)
ax.set_global()
ax.coastlines(lw=0.5, color="w")
ax.set_title("Same tessellation on a cartopy Robinson projection")
fig.savefig("example2_global_robinson", dpi=140, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# Example 3 -- REGIONAL tessellation clipped to a polygon (the Mediterranean
# region of the tutorial "Surface-Wave Tomography on the Sphere"), on a
# PlateCarree map with coastlines.
# ----------------------------------------------------------------------------
polygon = shapely.geometry.Polygon(
    [(-10, 30), (40, 30), (40, 40), (32, 47), (-2, 47), (-10, 40)]
)
med_sites = []
while len(med_sites) < 60:  # sites uniform per unit area within the polygon
    lon = rng.uniform(-10, 40)
    lat = np.degrees(
        np.arcsin(rng.uniform(np.sin(np.radians(30)), np.sin(np.radians(47))))
    )
    if polygon.contains(shapely.geometry.Point(lon, lat)):
        med_sites.append([lon, lat])
med_sites = np.array(med_sites)
med_values = 3 + 0.3 * np.sin(2 * np.pi * (med_sites[:, 0] + 6) / 14) * np.sin(
    2 * np.pi * (med_sites[:, 1] - 30) / 8
)

fig = plt.figure(figsize=(11, 4.5))
ax = plt.axes(projection=ccrs.PlateCarree())
Voronoi2DSphere.plot_tessellation(
    med_sites,
    med_values,
    ax=ax,
    clip_polygon=polygon,
    cmap=scm.roma,
    transform=ccrs.PlateCarree(),
)
ax.coastlines(resolution="50m", lw=0.7)
ax.set_extent([-11, 41, 29, 48], crs=ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True, lw=0.2)
gl.top_labels = gl.right_labels = False
ax.set_title("Regional tessellation clipped to the region of interest")
fig.savefig("example3_regional_clip", dpi=140, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# Example 4 -- Voronoi2D: default rendering vs the clip_polygon argument,
# same sites and values.
# ----------------------------------------------------------------------------
plane_sites = rng.uniform(0, 10, (150, 2))
plane_values = 3 + 0.3 * np.sin(plane_sites[:, 0]) * np.sin(plane_sites[:, 1])
region = shapely.geometry.Polygon([(1, 1), (9, 2), (8.5, 8.5), (5, 9.5), (1.5, 7)])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
Voronoi2D.plot_tessellation(plane_sites, plane_values, ax=axes[0], cmap=scm.roma)
axes[0].set_title("Voronoi2D, default")
Voronoi2D.plot_tessellation(
    plane_sites, plane_values, ax=axes[1], cmap=scm.roma, clip_polygon=region
)
axes[1].plot(*region.exterior.xy, "k", lw=1.2)
axes[1].set_title("Voronoi2D, clip_polygon=region")
fig.tight_layout()
fig.savefig("example4_voronoi2d", dpi=140, bbox_inches="tight")
plt.close(fig)

print("examples written to the working directory")
