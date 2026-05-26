import numpy as np

# WGS-84 Ellipsoid Constants
A = 6378137.0  # Semi-major axis in meters
F = 1.0 / 298.257223563  # Flattening
B = A * (1.0 - F)  # Semi-minor axis (~6356752.3142 m)
E2 = 2.0 * F - F ** 2  # Eccentricity squared
E_PRIME2 = (A ** 2 - B ** 2) / (B ** 2)  # Second eccentricity squared

def geodetic_to_ecef(lat, lon, alt):
    """
    Convert Geodetic coordinates (Latitude/Longitude in degrees, Altitude in meters)
    to ECEF (Earth-Centered, Earth-Fixed) Cartesian coordinates.
    Supports scalars and numpy arrays.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    n = A / np.sqrt(1.0 - E2 * sin_lat**2)
    
    x = (n + alt) * cos_lat * cos_lon
    y = (n + alt) * cos_lat * sin_lon
    z = (n * (1.0 - E2) + alt) * sin_lat
    
    return x, y, z

def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF Cartesian coordinates (in meters) to Geodetic coordinates
    (Latitude, Longitude in degrees, Altitude in meters) using Bowring's method.
    Supports scalars and numpy arrays.
    """
    p = np.sqrt(x**2 + y**2)
    
    # Avoid division by zero at the poles
    is_polar = p < 1e-6
    
    # Handle normal/polar points gracefully
    if np.isscalar(x):
        if is_polar:
            lat = 90.0 if z > 0 else -90.0
            lon = 0.0
            alt = np.abs(z) - B
            return lat, lon, alt
    else:
        # Array-based handling
        p = np.maximum(p, 1e-9)
        
    theta = np.arctan2(z * A, p * B)
    
    lat_rad = np.arctan2(
        z + E_PRIME2 * B * np.sin(theta)**3,
        p - E2 * A * np.cos(theta)**3
    )
    lon_rad = np.arctan2(y, x)
    
    n = A / np.sqrt(1.0 - E2 * np.sin(lat_rad)**2)
    alt = p / np.cos(lat_rad) - n
    
    lat = np.degrees(lat_rad)
    lon = np.degrees(lon_rad)
    
    if not np.isscalar(x):
        # Override values for polar coordinates in arrays
        lat = np.where(is_polar, np.where(z > 0, 90.0, -90.0), lat)
        lon = np.where(is_polar, 0.0, lon)
        alt = np.where(is_polar, np.abs(z) - B, alt)
        
    return lat, lon, alt

def get_local_normal(lat, lon):
    """
    Get the unit normal vector pointing to Zenith (upward) at a geodetic position.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    nx = np.cos(lat_rad) * np.cos(lon_rad)
    ny = np.cos(lat_rad) * np.sin(lon_rad)
    nz = np.sin(lat_rad)
    
    return np.array([nx, ny, nz])

def calculate_elevation_angle(rx_ecef, tx_ecef, rx_lat_lon=None):
    """
    Calculate the elevation angle (in degrees) of a transmitter (tx_ecef)
    as seen by a receiver (rx_ecef).
    """
    rx_ecef = np.array(rx_ecef)
    tx_ecef = np.array(tx_ecef)
    
    r = tx_ecef - rx_ecef
    d = np.linalg.norm(r, axis=-1)
    
    # Avoid division by zero
    if np.isscalar(d) and d < 1e-3:
        return 90.0
    elif not np.isscalar(d):
        d = np.maximum(d, 1e-3)
        
    if rx_lat_lon is None:
        rx_lat, rx_lon, _ = ecef_to_geodetic(rx_ecef[0], rx_ecef[1], rx_ecef[2])
    else:
        rx_lat, rx_lon = rx_lat_lon
        
    n = get_local_normal(rx_lat, rx_lon)
    
    # Dot product along the last axis
    dot_prod = np.sum(r * n, axis=-1)
    
    sin_el = dot_prod / d
    # Clip sin_el to prevent floating point out of bounds [-1, 1]
    sin_el = np.clip(sin_el, -1.0, 1.0)
    
    el_rad = np.arcsin(sin_el)
    return np.degrees(el_rad)

def radio_horizon_distance(h1, h2):
    """
    Calculate the maximum radio horizon line-of-sight distance (in meters)
    between height h1 (meters) and height h2 (meters) using 4/3 Earth radius model.
    """
    # d = sqrt(2 * R_eff * h)
    # R_eff = 4/3 * R_earth = 8494.67 km = 8494670 meters
    # coeff = sqrt(2 * 8494670) = 4121.81
    # We enforce h >= 0 to prevent complex numbers
    h1 = np.maximum(h1, 0.0)
    h2 = np.maximum(h2, 0.0)
    
    d1 = 4121.81 * np.sqrt(h1)
    d2 = 4121.81 * np.sqrt(h2)
    
    return d1 + d2
