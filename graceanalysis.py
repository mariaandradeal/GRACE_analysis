"""
GRACE-FO 2022 TWSA Analysis for India and Brazil
------------------------------------------------
Requirements:
    pip install xarray netCDF4 numpy pandas matplotlib cartopy

Data:
    Assumes JPL GRACE/GRACE-FO mascon product:
    GRCTellus.JPL.200204_202508.GLO.RL06.3M.MSCNv04CRI.nc

Outputs (publication style):
    - Map of 2022-mean TWSA with India & Brazil boxes
    - Regional pannel TWSA and monthly changes
    - Monthly maps for each box (India, Brazil)
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import MonthLocator, DateFormatter

# ------------------------- USER SETTINGS -----------------------------------

# >>> CHANGE THIS TO YOUR REAL PATH <<<
nc_path = r"C:\Users\maria\OneDrive - University of Arizona\Desktop\Pessoal\7. Pessoal\01. PHD\7. Remote sensing Hydrology\Assignment 11\GRCTellus.JPL.200204_202508.GLO.RL06.3M.MSCNv04CRI.nc"

# Variable names in the GRACE file (adjust if needed)
VAR_TWSA = "lwe_thickness"   # TWS anomalies (often in cm water equivalent)
VAR_SCALE = "scale_factor"   # Land/ocean scale factor (0–1)

# Analysis year
YEAR = 2022

# Region bounding boxes (lat_min, lat_max, lon_min, lon_max)
# Longitudes given in degrees East (−180 to 180); code will normalize to dataset convention.
regions = {
    "India": {
        "lat_bounds": (6.0, 36.0),
        "lon_bounds": (68.0, 98.0)
    },
    "Brazil": {
        "lat_bounds": (-35.0, 5.0),
        "lon_bounds": (-75.0, -34.0)
    }
}

# ------------------------- HELPER FUNCTIONS --------------------------------


def detect_lat_lon_names(ds):
    """Detect standard latitude and longitude coordinate names."""
    if "lat" in ds.coords:
        lat_name = "lat"
    elif "latitude" in ds.coords:
        lat_name = "latitude"
    else:
        raise ValueError("Could not find latitude coordinate in dataset.")

    if "lon" in ds.coords:
        lon_name = "lon"
    elif "longitude" in ds.coords:
        lon_name = "longitude"
    else:
        raise ValueError("Could not find longitude coordinate in dataset.")

    return lat_name, lon_name


def normalize_lon_bounds(lon_array, lon_min, lon_max):
    """
    Normalize requested lon bounds to match dataset convention:
    - If dataset is 0–360 and lon_min/max are negative, wrap them.
    """
    lon_min_norm, lon_max_norm = lon_min, lon_max
    lon_min_ds = float(lon_array.min())
    lon_max_ds = float(lon_array.max())

    # Dataset likely 0–360
    if lon_min_ds >= 0 and lon_max_ds > 180:
        if lon_min < 0:
            lon_min_norm = (lon_min + 360.0) % 360.0
        if lon_max < 0:
            lon_max_norm = (lon_max + 360.0) % 360.0

    return lon_min_norm, lon_max_norm


def select_region(da, lat_name, lon_name, lat_bounds, lon_bounds):
    """Subset DataArray to a lat/lon box, handling lat direction and lon wrap."""
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    # Normalize lon bounds to dataset convention
    lon_min, lon_max = normalize_lon_bounds(da[lon_name], lon_min, lon_max)

    # Latitude slice (dataset may be N→S or S→N)
    lat_vals = da[lat_name]
    if lat_vals[0] > lat_vals[-1]:
        lat_slice = slice(lat_max, lat_min)  # descending
    else:
        lat_slice = slice(lat_min, lat_max)  # ascending

    # Longitude slice (assumed ascending)
    lon_slice = slice(lon_min, lon_max)

    return da.sel({lat_name: lat_slice, lon_name: lon_slice})


def get_units_conversion_factor(units_str):
    """
    Convert GRACE units to cm water equivalent.
    Typical GRACE mascon units are already 'cm'.
    """
    units = units_str.lower()

    # Already cm water equivalent
    if "cm" in units:
        return 1.0

    # Millimeters of water
    if "mm" in units:
        return 0.1  # 10 mm = 1 cm

    # Meters of water
    if units.startswith("m") and "s" not in units:
        return 100.0  # 1 m = 100 cm

    # kg/m^2 ≈ mm of water
    if "kg" in units and "m-2" in units:
        return 0.1  # 1 kg/m² ≈ 1 mm ≈ 0.1 cm

    # Fallback: assume already cm
    return 1.0


def regional_timeseries(ds, var_name, scale_name, region_name, region_def):
    """
    Compute area-weighted, land-masked monthly TWSA timeseries for a region.
    Returns a pandas.Series indexed by time.
    """
    lat_name, lon_name = detect_lat_lon_names(ds)

    # Restrict to analysis year
    ds_year = ds.sel(time=slice(f"{YEAR}-01-01", f"{YEAR}-12-31"))
    twsa_year = ds_year[var_name]
    scale_da = ds[scale_name]

    # Unit conversion to cm
    units = twsa_year.attrs.get("units", "cm")
    factor = get_units_conversion_factor(units)
    twsa_cm = twsa_year * factor
    twsa_cm.attrs["units"] = "cm"

    # Subset both TWSA and scale factor to region
    region_twsa = select_region(
        twsa_cm, lat_name, lon_name,
        region_def["lat_bounds"], region_def["lon_bounds"]
    )
    region_scale = select_region(
        scale_da, lat_name, lon_name,
        region_def["lat_bounds"], region_def["lon_bounds"]
    )

    # Land mask from scale factor: 1 over land, 0 over ocean (NO NaNs here)
    land_mask = xr.where(region_scale > 0, 1.0, 0.0)

    # Apply scale factor to TWSA (leakage correction / land emphasis)
    region_twsa_scaled = region_twsa * region_scale

    # Area weights ~ cos(latitude)
    coslat = np.cos(np.deg2rad(region_twsa_scaled[lat_name]))
    # Broadcast coslat to full 2D lat-lon and apply land mask
    weights = coslat * land_mask

    # Remove any residual NaNs from weights
    weights = weights.fillna(0)

    # Area-weighted mean over land
    ts = region_twsa_scaled.weighted(weights).mean(dim=(lat_name, lon_name))

    # Convert to pandas.Series for easy plotting
    ts_series = ts.to_series()
    ts_series.name = f"TWSA_{region_name}"

    return ts_series


def compute_monthly_change(ts_series):
    """Compute month-to-month change ΔTWSA (current − previous)."""
    diff_series = ts_series.diff()
    diff_series.name = ts_series.name + "_dT"
    return diff_series


def plot_monthly_maps_for_region(ds, var_name, scale_name, region_name, region_def):
    """
    Improved publication-style monthly panel plot:
    - Bigger vertical spacing
    - Month titles repositioned upward
    - Clean axes, fixed colorbar
    """
    lat_name, lon_name = detect_lat_lon_names(ds)

    # Restrict to analysis year and convert units
    ds_year = ds.sel(time=slice(f"{YEAR}-01-01", f"{YEAR}-12-31"))
    twsa_year = ds_year[var_name]
    units = twsa_year.attrs.get("units", "cm")
    factor = get_units_conversion_factor(units)
    twsa_cm = twsa_year * factor

    # Subset region and mask to land
    region_twsa = select_region(
        twsa_cm, lat_name, lon_name,
        region_def["lat_bounds"], region_def["lon_bounds"]
    )
    region_scale = select_region(
        ds[scale_name], lat_name, lon_name,
        region_def["lat_bounds"], region_def["lon_bounds"]
    )
    region_twsa = region_twsa.where(region_scale > 0)

    times = pd.to_datetime(region_twsa["time"].values)
    ntime = len(times)

    # Layout: 3 rows x 4 columns (12 months)
    ncols = 4
    nrows = int(np.ceil(ntime / ncols))

    # Fixed color range
    vmin, vmax = -30.0, 30.0

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.0 * ncols, 3.3 * nrows),     # slightly taller panels
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    axes = np.array(axes).ravel()

    # Increased vertical spacing (key fix!)
    fig.subplots_adjust(
        left=0.04, right=0.98,
        top=0.90, bottom=0.15,
        wspace=0.03,
        hspace=0.20
    )

    # Extent
    lat_min, lat_max = region_def["lat_bounds"]
    lon_min, lon_max = region_def["lon_bounds"]
    lon_min_n, lon_max_n = normalize_lon_bounds(ds[lon_name], lon_min, lon_max)

    pcm = None
    for i, t in enumerate(times):
        ax = axes[i]
        da_t = region_twsa.isel(time=i)

        pcm = ax.pcolormesh(
            da_t[lon_name], da_t[lat_name], da_t,
            transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax, shading="auto",
            cmap="RdBu_r"
        )

        ax.coastlines(linewidth=0.4)
        ax.set_extent([lon_min_n, lon_max_n, lat_min, lat_max], ccrs.PlateCarree())

        # Clean axes
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

        # ------------------------------
        # Move title UP inside the panel
        # ------------------------------
        ax.text(
            0.02, 1.05,
            t.strftime("%b"),
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom"
        )

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Colorbar
    cax = fig.add_axes([0.18, 0.06, 0.64, 0.025])
    cbar = fig.colorbar(pcm, cax=cax, orientation="horizontal")
    cbar.set_label("GRACE TWSA (cm)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(f"{region_name} – Monthly GRACE TWSA (cm), {YEAR}", fontsize=14, y=0.96)

    fig.savefig(f"GRACE_TWSA_{region_name}_monthly_maps_{YEAR}.pdf", dpi=300, bbox_inches="tight")


# ------------------------- MAIN ANALYSIS -----------------------------------


def main():
    # Open dataset
    ds = xr.open_dataset(nc_path, engine="netcdf4")

    lat_name, lon_name = detect_lat_lon_names(ds)

    # Pre-compute regional time series
    ts_regions = {}
    dts_regions = {}

    for rname, rdef in regions.items():
        ts = regional_timeseries(ds, VAR_TWSA, VAR_SCALE, rname, rdef)
        dts = compute_monthly_change(ts)
        ts_regions[rname] = ts
        dts_regions[rname] = dts

    # ------------------------- PLOTTING: MAP WITH BOXES ------------------------

    # Compute 2022 mean TWSA (cm) for display
    twsa = ds[VAR_TWSA].sel(time=slice(f"{YEAR}-01-01", f"{YEAR}-12-31"))
    units = twsa.attrs.get("units", "cm")
    factor = get_units_conversion_factor(units)
    twsa_cm = (twsa * factor).mean("time")

    # Mask ocean using scale factor so map shows land signal only
    scale = ds[VAR_SCALE]
    twsa_cm = twsa_cm.where(scale > 0)

    # Fix color range to -30 .. +30 cm
    vmin, vmax = -30.0, 30.0

    fig = plt.figure(figsize=(10, 4.5))
    proj = ccrs.Robinson()
    ax = plt.subplot(1, 1, 1, projection=proj)

    lon2d, lat2d = np.meshgrid(ds[lon_name], ds[lat_name])
    pcm = ax.pcolormesh(
        lon2d,
        lat2d,
        twsa_cm,
        transform=ccrs.PlateCarree(),
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu_r"  # diverging, anomalies around zero
    )

    ax.coastlines(linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.set_global()

    cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05)
    cbar.set_label("GRACE TWSA (cm), 2022 mean", fontsize=11)

    # Draw region boxes
    for rname, rdef in regions.items():
        lat_min, lat_max = rdef["lat_bounds"]
        lon_min, lon_max = rdef["lon_bounds"]
        lon_min_n, lon_max_n = normalize_lon_bounds(ds[lon_name], lon_min, lon_max)

        # Box corners
        lats_box = [lat_min, lat_max, lat_max, lat_min, lat_min]
        lons_box = [lon_min_n, lon_min_n, lon_max_n, lon_max_n, lon_min_n]

        ax.plot(
            lons_box,
            lats_box,
            transform=ccrs.PlateCarree(),
            linewidth=1.5,
            linestyle="-",
        )

        # Put label near box center
        ax.text(
            (lon_min_n + lon_max_n) / 2.0,
            (lat_min + lat_max) / 2.0,
            rname,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_title(f"GRACE-FO Mascon TWSA (2022 mean) with India & Brazil boxes", fontsize=12)
    plt.tight_layout()
    plt.savefig("GRACE_TWSA_2022_map_boxes.pdf", dpi=300, bbox_inches="tight")
    # plt.show()

    # ------------------------- PLOTTING: 2×2 TIME SERIES -----------------------

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9
    })

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    axes = axes.ravel()

    region_order = ["India", "Brazil"]

    for i, rname in enumerate(region_order):
        ts = ts_regions[rname]
        dts = dts_regions[rname]

        # Clean NaNs (diff will have NaN at first month)
        ts = ts.dropna()
        dts = dts.dropna()

        # (a) TWSA time series
        ax_ts = axes[i * 2]
        ax_ts.plot(ts.index, ts.values, marker="o", linewidth=1.5)
        ax_ts.grid(True, linestyle=":", linewidth=0.5)
        ax_ts.set_ylabel("TWSA (cm)")
        ax_ts.set_title(f"{rname} – Monthly mean TWSA ({YEAR})")

        ax_ts.xaxis.set_major_locator(MonthLocator())
        ax_ts.xaxis.set_major_formatter(DateFormatter("%b"))
        ax_ts.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

        # (b) Month-to-month change
        ax_dts = axes[i * 2 + 1]
        ax_dts.bar(dts.index, dts.values, width=20)  # ~20 days width just for visual
        ax_dts.axhline(0, color="k", linewidth=0.8)
        ax_dts.grid(True, linestyle=":", linewidth=0.5, axis="y")
        ax_dts.set_ylabel("ΔTWSA (cm)")
        ax_dts.set_title(f"{rname} – Month-to-month change ({YEAR})")

        ax_dts.xaxis.set_major_locator(MonthLocator())
        ax_dts.xaxis.set_major_formatter(DateFormatter("%b"))
        ax_dts.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

    # Rotate x-tick labels (if needed)
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(0)

    #fig.suptitle(f"GRACE-FO Regional TWSA and Monthly Changes – {YEAR}", y=0.99, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("GRACE_TWSA_India_Brazil_timeseries.pdf", dpi=300, bbox_inches="tight")
    # plt.show()

    # ------------------------- MONTHLY MAPS FOR EACH BOX -----------------------

    for rname, rdef in regions.items():
        plot_monthly_maps_for_region(ds, VAR_TWSA, VAR_SCALE, rname, rdef)


if __name__ == "__main__":
    main()

