import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean
import pandas as pd
from matplotlib.colors import BoundaryNorm
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os
from datetime import datetime

@dataclass
class LayerScheme:
    """Class to store layer scheme information"""
    name: str
    thicknesses: np.ndarray
    bottoms: np.ndarray
    middles: np.ndarray
    n_surface: int  # number of fixed surface layers
    n_root: int     # number of root zone layers
    n_deep: int     # number of deep soil layers

class HybridSoilLayerGenerator:
    def __init__(self):
        self.schemes = {}
        
    def generate_hybrid_layers(self,
                             surface_layers: List[float] = [0.05, 0.10],
                             n_root_layers: int = 4,
                             n_deep_layers: int = 4,
                             root_growth_factor: float = 1.6,
                             deep_growth_factor: float = 1.7,
                             bedrock_depth: float = None,
                             target_root_depth: float = 1.0) -> np.ndarray:
        """
        Generate layer thicknesses using a hybrid approach:
        1. Fixed thin layers near surface for satellite validation
        2. Geometric progression in root zone
        3. Steeper geometric progression in deep soil
        """
        if bedrock_depth is None:
            bedrock_depth = 2 * target_root_depth
        
        # 1. Surface layers
        thicknesses = np.array(surface_layers)
        surface_depth = np.sum(thicknesses)
        
        # 2. Root zone layers
        remaining_root_depth = target_root_depth - surface_depth
        if remaining_root_depth > 0 and n_root_layers > 0:
            initial_root_thickness = max(
                surface_layers[-1] * 1.2,
                remaining_root_depth * (root_growth_factor - 1) / (root_growth_factor ** n_root_layers - 1)
            )
            root_thicknesses = initial_root_thickness * root_growth_factor ** np.arange(n_root_layers)
            thicknesses = np.concatenate([thicknesses, root_thicknesses])
        
        # 3. Deep soil layers
        if bedrock_depth > target_root_depth and n_deep_layers > 0:
            remaining_depth = bedrock_depth - np.sum(thicknesses)
            initial_deep_thickness = max(
                root_thicknesses[-1] * 1.2 if n_root_layers > 0 else surface_layers[-1] * 1.2,
                remaining_depth * (deep_growth_factor - 1) / (deep_growth_factor ** n_deep_layers - 1)
            )
            deep_thicknesses = initial_deep_thickness * deep_growth_factor ** np.arange(n_deep_layers)
            thicknesses = np.concatenate([thicknesses, deep_thicknesses])
        
        return thicknesses

    def create_scheme(self, name: str, **kwargs) -> LayerScheme:
        """Create a layer scheme with the given parameters"""
        thicknesses = self.generate_hybrid_layers(**kwargs)
        bottoms = np.cumsum(thicknesses)
        middles = bottoms - thicknesses/2
        
        n_surface = len(kwargs.get('surface_layers', [0.05, 0.10]))
        n_root = kwargs.get('n_root_layers', 4)
        n_deep = kwargs.get('n_deep_layers', 4)
        
        scheme = LayerScheme(
            name=name,
            thicknesses=thicknesses,
            bottoms=bottoms,
            middles=middles,
            n_surface=n_surface,
            n_root=n_root,
            n_deep=n_deep
        )
        
        self.schemes[name] = scheme
        return scheme

    def plot_schemes(self, 
                    scheme_names: List[str] = None,
                    figsize: Tuple[float, float] = (20, 10),
                    show_zones: bool = True,
                    save_path: str = None,
                    dpi: int = 300) -> None:
        """Create visualization of soil layer schemes with annotations"""
        if scheme_names is None:
            scheme_names = list(self.schemes.keys())
            
        n_schemes = len(scheme_names)
        fig, axes = plt.subplots(1, n_schemes, figsize=figsize)
        if n_schemes == 1:
            axes = [axes]
            
        zone_colors = ['#ffe5e5', '#e5ffe5', '#e5e5ff']
        zone_labels = ['Surface', 'Root Zone', 'Deep Soil']
        
        key_depths = {
            0.05: 'Satellite validation depth (5cm)',
            0.30: 'STATSGO2/HWSD data boundary (30cm)',
            1.00: 'Primary root zone (1m)',
            2.00: 'Typical GCM minimum depth (2m)',
            3.50: 'Original model depth (3.5m)'
        }
        
        for ax, name in zip(axes, scheme_names):
            scheme = self.schemes[name]
            
            if show_zones:
                surface_depth = scheme.bottoms[scheme.n_surface - 1]
                ax.axhspan(0, surface_depth, color=zone_colors[0], alpha=0.3)
                
                root_depth = scheme.bottoms[scheme.n_surface + scheme.n_root - 1]
                ax.axhspan(surface_depth, root_depth, color=zone_colors[1], alpha=0.3)
                
                ax.axhspan(root_depth, scheme.bottoms[-1], color=zone_colors[2], alpha=0.3)
            
            for i, (top, bottom) in enumerate(zip(
                np.concatenate(([0], scheme.bottoms[:-1])),
                scheme.bottoms
            )):
                if i < scheme.n_surface:
                    zone_idx = 0
                elif i < scheme.n_surface + scheme.n_root:
                    zone_idx = 1
                else:
                    zone_idx = 2
                    
                ax.fill_betweenx([top, bottom], 0, 1,
                                alpha=0.3,
                                color=plt.cm.viridis(i/len(scheme.thicknesses)))
                
                mid = (top + bottom) / 2
                ax.text(0.5, mid, f'{scheme.thicknesses[i]:.3f}m',
                       ha='center', va='center', fontsize=8)
            
            for depth, label in key_depths.items():
                if depth <= scheme.bottoms[-1]:
                    ax.axhline(y=depth, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
                    
                    if ax == axes[0]:
                        ax.text(-0.5, depth, label, 
                               va='center', ha='right', fontsize=8,
                               bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, pad=2))
            
            ax.set_ylim(scheme.bottoms[-1], 0)
            ax.set_xlim(0, 1)
            ax.set_title(f'{name}\nTotal depth: {scheme.bottoms[-1]:.2f}m\n{len(scheme.thicknesses)} layers',
                        pad=20)
            ax.set_xticks([])
            ax.set_ylabel('Depth (m)')
            
            if show_zones and ax == axes[0]:
                for i, (label, color) in enumerate(zip(zone_labels, zone_colors)):
                    ax.text(-0.3, scheme.bottoms[min(
                        scheme.n_surface - 1 if i == 0 else
                        scheme.n_surface + scheme.n_root - 1 if i == 1 else
                        len(scheme.bottoms) - 1,
                        len(scheme.bottoms) - 1
                    )] / 2, label,
                           rotation=90, va='center', ha='center',
                           color='k', bbox=dict(facecolor=color, alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                       exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            print(f"Figure saved to: {save_path}")
            
        plt.show()

class SoilDataAnalyzer:
    def __init__(self, filepath):
        """Initialize with NetCDF file path"""
        self.ds = xr.open_dataset(filepath)
        self.components = ['Sand', 'Silt', 'Clay', 'Peat', 'Bedrock']
        
    def plot_soil_components(self, layer=0, save_path=None, dpi=300):
        """
        Create a multi-panel plot showing distribution of soil components
        with standardized colorbars and highlighted unphysical values
        """
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.1)
        
        panel_positions = [
            (0, 0), (0, 1),  # Sand, Silt (top row)
            (1, 0), (1, 1),  # Clay, Peat (middle row)
            (2, 0), None     # Bedrock (bottom left)
        ]
        
        projection = ccrs.Robinson(central_longitude=0)
        axes = []
        levels = np.linspace(0, 1, 11)
        
        for idx, pos in enumerate(panel_positions):
            if pos is not None and idx < 5:
                ax = fig.add_subplot(gs[pos], projection=projection)
                axes.append(ax)
        
        for idx, (ax, component) in enumerate(zip(axes, self.components)):
            data = self.ds.q.isel(ngm=layer, imt=idx)
            
            if component == 'Sand':
                cmap = cmocean.cm.haline_r
            elif component == 'Silt':
                cmap = cmocean.cm.dense
            elif component == 'Clay':
                cmap = cmocean.cm.turbid
            elif component == 'Peat':
                cmap = cmocean.cm.algae
            else:  # Bedrock
                cmap = cmocean.cm.gray_r  # Reversed colormap for bedrock
            
            physical_data = np.ma.masked_where(
                (data < 0) | (data > 1), 
                data
            )
            unphysical_data = np.ma.masked_where(
                (data >= 0) & (data <= 1), 
                data
            )
            
            img = ax.contourf(self.ds.lon, self.ds.lat, physical_data,
                            transform=ccrs.PlateCarree(),
                            cmap=cmap, levels=levels, extend='both')
            
            if not np.all(unphysical_data.mask):
                ax.contourf(self.ds.lon, self.ds.lat, unphysical_data,
                          transform=ccrs.PlateCarree(),
                          colors=['red'], alpha=0.5)
            
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k', facecolor='white')
            
            depth = self.ds.dz.isel(ngm=layer).mean().values
            ax.set_title(f'{component} Content - Layer {layer+1} (≈{depth:.2f}m)', 
                        fontsize=12, pad=10)
            
            cb = plt.colorbar(img, ax=ax, orientation='horizontal',
                            pad=0.05, fraction=0.05,
                            ticks=levels[::2])
            cb.set_label(f'{component} Fraction', fontsize=10)
            
            if not np.all(unphysical_data.mask):
                unphys_min = np.min(unphysical_data.compressed())
                unphys_max = np.max(unphysical_data.compressed())
                ax.text(0.02, 0.02, f'Unphysical values: [{unphys_min:.2f}, {unphys_max:.2f}]',
                       transform=ax.transAxes, color='red', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7))
        
        plt.suptitle(f'Global Soil Component Distribution\nLayer {layer+1} (Depth ≈ {depth:.2f}m)',
                    fontsize=14, y=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved figure to: {save_path}")
        
        plt.show()
    
    def interpolate_to_scheme(self, scheme: LayerScheme) -> xr.Dataset:
        """Interpolate soil data to a new layer scheme"""
        orig_depths = self.ds.dz.mean(dim=['lat', 'lon']).values
        new_depths = scheme.bottoms
        
        new_ds = self.ds.copy()
        
        for imt in range(5):
            q_vals = self.ds.q.isel(imt=imt)
            qk_vals = self.ds.qk.isel(imt=imt)
            
            new_q = np.zeros((len(new_depths), len(self.ds.lat), len(self.ds.lon)))
            new_qk = np.zeros_like(new_q)
            
            for lat in range(len(self.ds.lat)):
                for lon in range(len(self.ds.lon)):
                    new_q[:, lat, lon] = np.interp(new_depths, 
                                                 orig_depths, 
                                                 q_vals[:, lat, lon])
                    new_qk[:, lat, lon] = np.interp(new_depths,
                                                  orig_depths,
                                                  qk_vals[:, lat, lon])
            
            new_ds.q.loc[dict(imt=imt)] = new_q
            new_ds.qk.loc[dict(imt=imt)] = new_qk
        
        new_ds.dz.values = np.tile(scheme.thicknesses, 
                                 (len(self.ds.lat), len(self.ds.lon), 1)).transpose(2, 0, 1)
        
        return new_ds

def main():
    # Initialize analyzer
    analyzer = SoilDataAnalyzer("S144X900098M.ext.nc")
    
    # Create vertical layer scheme generator
    generator = HybridSoilLayerGenerator()
    
    # Create satellite-optimized scheme
    sat_scheme = generator.create_scheme(
        "Satellite-Optimized",
        surface_layers=[0.05, 0.10],    # Fixed surface layers (5cm for satellite)
        n_root_layers=4,                # Root zone layers
        n_deep_layers=3,                # Deep soil layers
        root_growth_factor=1.6,         # More gradual progression
        deep_growth_factor=1.7,         # Steeper increase with depth
        bedrock_depth=3.5               # Match original depth
    )
    
    # Plot original soil components
    print("Plotting original soil components...")
    for layer in range(len(analyzer.ds.dz)):
        analyzer.plot_soil_components(
            layer=layer,
            save_path=f"original_soil_maps_layer_{layer+1}.png",
            dpi=300
        )
    
    # Plot layer scheme comparison
    generator.plot_schemes(
        figsize=(15, 8),
        save_path="soil_schemes_comparison.png",
        dpi=300
    )
    
    # Interpolate to new scheme
    print("\nInterpolating to new layer scheme...")
    new_ds = analyzer.interpolate_to_scheme(sat_scheme)
    
    # Create temporary analyzer for plotting new data
    print("\nPlotting interpolated soil components...")
    temp_analyzer = SoilDataAnalyzer(None)
    temp_analyzer.ds = new_ds
    for layer in range(len(new_ds.dz)):
        temp_analyzer.plot_soil_components(
            layer=layer,
            save_path=f"interpolated_soil_maps_layer_{layer+1}.png",
            dpi=300
        )
    
    # Save new dataset
    output_filename = "soil_data_satellite_optimized.nc"
    print(f"\nSaving interpolated dataset to: {output_filename}")
    new_ds.to_netcdf(output_filename)
    print("Done!")

if __name__ == "__main__":
    main()