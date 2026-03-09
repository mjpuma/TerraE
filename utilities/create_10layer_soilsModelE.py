import xarray as xr
import numpy as np

def create_10layer_soil_file(input_file: str, output_file: str):
    """
    Create a 10-layer soil file matching exact format of original file.
    """
    # Read original dataset
    ds = xr.open_dataset(input_file)
    
    # Define new layer depths (bottom depths of each layer)
    new_depths = np.array([0.05, 0.15, 0.30, 0.50, 0.80, 1.20, 1.80, 2.50, 3.50, 5.00])
    new_thicknesses = np.diff(np.concatenate(([0], new_depths)))
    
    # Calculate original and new layer depths for interpolation
    orig_thicknesses = ds.dz.isel(lat=0, lon=0).values
    orig_bottoms = np.cumsum(orig_thicknesses)
    orig_tops = np.concatenate(([0], orig_bottoms[:-1]))
    orig_mids = (orig_tops + orig_bottoms) / 2
    new_tops = np.concatenate(([0], new_depths[:-1]))
    new_mids = (new_tops + new_depths) / 2
    
    # Initialize arrays
    new_q = np.zeros((10, 5, 90, 144), dtype=np.float32)
    new_qk = np.zeros_like(new_q)
    
    # Interpolate data
    for imt in range(5):
        for lat in range(90):
            for lon in range(144):
                orig_q = ds.q.isel(imt=imt, lat=lat, lon=lon).values
                orig_qk = ds.qk.isel(imt=imt, lat=lat, lon=lon).values
                
                new_q[:, imt, lat, lon] = np.interp(
                    new_mids,
                    orig_mids,
                    orig_q,
                    left=orig_q[0],
                    right=orig_q[-1]
                )
                
                new_qk[:, imt, lat, lon] = np.interp(
                    new_mids,
                    orig_mids,
                    orig_qk,
                    left=orig_qk[0],
                    right=orig_qk[-1]
                )
    
    # Normalize fractions
    for layer in range(10):
        layer_sum = np.sum(new_q[layer, :, :, :], axis=0)
        for imt in range(5):
            new_q[layer, imt, :, :] = new_q[layer, imt, :, :] / layer_sum
            new_qk[layer, imt, :, :] = new_qk[layer, imt, :, :] / layer_sum
    
    # Create main dataset without sl
    ds_main = xr.Dataset(
        {
            'lon': (['lon'], ds.lon.values, {'units': 'degrees_east'}),
            'lat': (['lat'], ds.lat.values, {'units': 'degrees_north'}),
            'dz': (['ngm', 'lat', 'lon'], 
                  np.tile(new_thicknesses, (144, 90, 1)).transpose(2, 1, 0).astype(np.float32)),
            'q': (['ngm', 'imt', 'lat', 'lon'], new_q.astype(np.float32)),
            'qk': (['ngm', 'imt', 'lat', 'lon'], new_qk.astype(np.float32))
        }
    )
    
    # Save main dataset first
    encoding_main = {var: {'_FillValue': None} for var in ds_main.variables}
    ds_main.to_netcdf(output_file, encoding=encoding_main)
    
    # Create final dataset with sl
    final_ds = xr.Dataset(
        {
            'lon': (['lon'], ds.lon.values, {'units': 'degrees_east'}),
            'lat': (['lat'], ds.lat.values, {'units': 'degrees_north'}),
            'dz': (['ngm', 'lat', 'lon'], 
                  np.tile(new_thicknesses, (144, 90, 1)).transpose(2, 1, 0).astype(np.float32)),
            'q': (['ngm', 'imt', 'lat', 'lon'], new_q.astype(np.float32)),
            'qk': (['ngm', 'imt', 'lat', 'lon'], new_qk.astype(np.float32)),
            'sl': (['lat', 'lon'], ds.sl.values)
        }
    )
    
    # Save final dataset with sl
    encoding_final = {var: {'_FillValue': None} for var in final_ds.variables}
    final_ds.to_netcdf(output_file, encoding=encoding_final)
    
    print(f"Created new 10-layer file: {output_file}")
    return final_ds

def main():
    input_file = "S144X900098M.ext.nc"
    output_file = "S144X900098M10layer.ext.nc"
    
    new_ds = create_10layer_soil_file(input_file, output_file)
    
    # Verify the structure
    print("\nNew dataset structure:")
    ds = xr.open_dataset(output_file)
    print(ds)
    
if __name__ == "__main__":
    main()