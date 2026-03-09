import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
                             surface_layers: List[float] = [0.05, 0.10],  # Fixed surface layers
                             n_root_layers: int = 4,                      # Number of root zone layers
                             n_deep_layers: int = 4,                      # Number of deep soil layers
                             root_growth_factor: float = 1.6,            # Growth factor for root zone
                             deep_growth_factor: float = 1.7,            # Growth factor for deep soil
                             bedrock_depth: float = None,                # Local bedrock depth
                             target_root_depth: float = 1.0) -> np.ndarray:     # Target depth for root zone
        """
        Generate layer thicknesses using a hybrid approach:
        1. Fixed thin layers near surface for satellite validation and rapid processes
        2. Geometric progression in root zone, starting thicker than surface layers
        3. Steeper geometric progression in deep soil
        """
        # Set default bedrock depth if not provided
        if bedrock_depth is None:
            bedrock_depth = 2 * target_root_depth
        
        # 1. Surface layers (fixed thicknesses)
        thicknesses = np.array(surface_layers)
        surface_depth = np.sum(thicknesses)
        
        # 2. Root zone layers (moderate geometric progression)
        remaining_root_depth = target_root_depth - surface_depth
        if remaining_root_depth > 0 and n_root_layers > 0:
            # Ensure initial root zone thickness is larger than last surface layer
            initial_root_thickness = max(
                surface_layers[-1] * 1.2,  # At least 20% thicker than last surface layer
                remaining_root_depth * (root_growth_factor - 1) / (root_growth_factor ** n_root_layers - 1)
            )
            root_thicknesses = initial_root_thickness * root_growth_factor ** np.arange(n_root_layers)
            thicknesses = np.concatenate([thicknesses, root_thicknesses])
        
        # 3. Deep soil layers (steeper geometric progression)
        if bedrock_depth > target_root_depth and n_deep_layers > 0:
            remaining_depth = bedrock_depth - np.sum(thicknesses)
            # Ensure initial deep thickness is larger than last root zone thickness
            initial_deep_thickness = max(
                root_thicknesses[-1] * 1.2 if n_root_layers > 0 else surface_layers[-1] * 1.2,
                remaining_depth * (deep_growth_factor - 1) / (deep_growth_factor ** n_deep_layers - 1)
            )
            deep_thicknesses = initial_deep_thickness * deep_growth_factor ** np.arange(n_deep_layers)
            thicknesses = np.concatenate([thicknesses, deep_thicknesses])
        
        return thicknesses

    def create_scheme(self, 
                     name: str,
                     **kwargs) -> LayerScheme:
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
        """
        Create visualization of soil layer schemes with annotations for key depths
        """
        if scheme_names is None:
            scheme_names = list(self.schemes.keys())
            
        n_schemes = len(scheme_names)
        fig, axes = plt.subplots(1, n_schemes, figsize=figsize)
        if n_schemes == 1:
            axes = [axes]
            
        zone_colors = ['#ffe5e5', '#e5ffe5', '#e5e5ff']  # Colors for different zones
        zone_labels = ['Surface', 'Root Zone', 'Deep Soil']
        
        # Define key depths and their labels
        key_depths = {
            0.05: 'Satellite validation depth (5cm)',
            0.30: 'STATSGO2/HWSD data boundary (30cm)',
            1.00: 'Primary root zone (1m)',
            2.00: 'Typical GCM minimum depth (2m)',
            3.50: 'Original model depth (3.5m)'
        }
        
        for ax, name in zip(axes, scheme_names):
            scheme = self.schemes[name]
            
            # Plot zones if requested
            if show_zones:
                # Surface zone
                surface_depth = scheme.bottoms[scheme.n_surface - 1]
                ax.axhspan(0, surface_depth, color=zone_colors[0], alpha=0.3)
                
                # Root zone
                root_depth = scheme.bottoms[scheme.n_surface + scheme.n_root - 1]
                ax.axhspan(surface_depth, root_depth, color=zone_colors[1], alpha=0.3)
                
                # Deep soil zone
                ax.axhspan(root_depth, scheme.bottoms[-1], color=zone_colors[2], alpha=0.3)
            
            # Plot layers
            for i, (top, bottom) in enumerate(zip(
                np.concatenate(([0], scheme.bottoms[:-1])),
                scheme.bottoms
            )):
                # Determine zone for coloring
                if i < scheme.n_surface:
                    zone_idx = 0
                elif i < scheme.n_surface + scheme.n_root:
                    zone_idx = 1
                else:
                    zone_idx = 2
                    
                ax.fill_betweenx([top, bottom], 0, 1,
                                alpha=0.3,
                                color=plt.cm.viridis(i/len(scheme.thicknesses)))
                
                # Add thickness label
                mid = (top + bottom) / 2
                ax.text(0.5, mid, f'{scheme.thicknesses[i]:.3f}m',
                       ha='center', va='center', fontsize=8)
            
            # Add horizontal lines and annotations for key depths
            for depth, label in key_depths.items():
                if depth <= scheme.bottoms[-1]:
                    # Add dashed line
                    ax.axhline(y=depth, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
                    
                    # Add annotation for first plot only
                    if ax == axes[0]:
                        ax.text(-0.5, depth, label, 
                               va='center', ha='right', fontsize=8,
                               bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, pad=2))
            
            ax.set_ylim(scheme.bottoms[-1], 0)  # Reverse y-axis
            ax.set_xlim(0, 1)
            ax.set_title(f'{name}\nTotal depth: {scheme.bottoms[-1]:.2f}m\n{len(scheme.thicknesses)} layers',
                        pad=20)
            ax.set_xticks([])
            ax.set_ylabel('Depth (m)')
            
            # Add zone labels if showing zones
            if show_zones and ax == axes[0]:  # Only for first plot
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
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                       exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            print(f"Figure saved to: {save_path}")
            
        plt.show()

    def save_to_csv(self, 
                    output_dir: str = "soil_schemes",
                    prefix: str = "") -> None:
        """Save all schemes to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, scheme in self.schemes.items():
            filename = f"{prefix}_{name.replace(' ', '_')}_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            
            df = pd.DataFrame({
                'Layer': range(1, len(scheme.thicknesses) + 1),
                'Thickness (m)': scheme.thicknesses,
                'Bottom Depth (m)': scheme.bottoms,
                'Middle Depth (m)': scheme.middles,
                'Zone': ['Surface'] * scheme.n_surface +
                       ['Root Zone'] * scheme.n_root +
                       ['Deep Soil'] * scheme.n_deep
            })
            df.to_csv(filepath, index=False)
            print(f"Saved scheme '{name}' to: {filepath}")


def main():
    generator = HybridSoilLayerGenerator()
    
    # Original 6-layer model
    generator.create_scheme(
        "Original 6-Layer Model",
        surface_layers=[0.100],            # First layer
        n_root_layers=3,                   # Next 3 layers
        n_deep_layers=2,                   # Final 2 layers
        root_growth_factor=1.73,           # Approximates original progression
        deep_growth_factor=1.75,           # Approximates original progression
        bedrock_depth=3.5                  # Original total depth
    )
    
    # Original 10-layer model
    generator.create_scheme(
        "Original 10-Layer Model",
        surface_layers=[0.050, 0.090],    # First two layers
        n_root_layers=4,                  # Next 4 layers
        n_deep_layers=4,                  # Final 4 layers
        root_growth_factor=1.8,          # Approximates original progression
        deep_growth_factor=1.85,         # Approximates original progression
        bedrock_depth=21.5               # Original total depth
    )
    
    # New satellite-optimized scheme
    generator.create_scheme(
        "Satellite-Optimized",
        surface_layers=[0.05, 0.10],       # Fixed surface layers (5cm for satellite)
        n_root_layers=4,                   # Root zone layers
        n_deep_layers=4,                   # Deep soil layers
        root_growth_factor=1.6,            # More gradual progression
        deep_growth_factor=1.7,            # Steeper increase with depth
        bedrock_depth=3.5                  # Match original depth
    )
    
    # Plot all schemes with annotations
    generator.plot_schemes(
        figsize=(20, 10),
        save_path="soil_schemes/model_comparison.png",
        dpi=300
    )
    
    # Save to CSV
    generator.save_to_csv(prefix="soil_schemes")

if __name__ == "__main__":
    main()