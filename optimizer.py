import numpy as np
import requests
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline
from coord_math import geodetic_to_ecef, ecef_to_geodetic, get_local_normal, radio_horizon_distance

def enu_to_ecef_offsets(enu_km, ref_ecef):
    """Convert local ENU offsets in kilometers to ECEF absolute coordinates in meters."""
    ref_lat, ref_lon, _ = ecef_to_geodetic(ref_ecef[0], ref_ecef[1], ref_ecef[2])
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)
    
    e = enu_km[0] * 1000.0
    n = enu_km[1] * 1000.0
    u = enu_km[2] * 1000.0
    
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u
    
    return np.array([ref_ecef[0] + dx, ref_ecef[1] + dy, ref_ecef[2] + dz])

def ecef_to_enu_offsets(ecef, ref_ecef):
    """Convert absolute ECEF coordinates in meters to local ENU offsets in kilometers."""
    ref_lat, ref_lon, _ = ecef_to_geodetic(ref_ecef[0], ref_ecef[1], ref_ecef[2])
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)
    
    dx = ecef[0] - ref_ecef[0]
    dy = ecef[1] - ref_ecef[1]
    dz = ecef[2] - ref_ecef[2]
    
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    
    return np.array([e, n, u]) / 1000.0

def compute_objective_enu(params, ref_ecef, tracks, horizon_weight=100.0):
    """
    Wrapper for compute_objective using local ENU coordinates in kilometers,
    with self-calibration of path loss exponent (eta) and antenna gain roll-off exponent (n_ant).
    """
    e_km, n_km, u_km, eta, n_ant = params
    
    # Calculate altitude bounds relative to reference altitude
    ref_lat, ref_lon, ref_alt = ecef_to_geodetic(ref_ecef[0], ref_ecef[1], ref_ecef[2])
    min_u_km = (0.0 - ref_alt) / 1000.0
    max_u_km = (2500.0 - ref_alt) / 1000.0
    
    penalty = 0.0
    # Horizontal bounds: +/- 200 km
    if e_km < -200.0: penalty += (e_km + 200.0)**2 * 1e5
    elif e_km > 200.0: penalty += (e_km - 200.0)**2 * 1e5
    
    if n_km < -200.0: penalty += (n_km + 200.0)**2 * 1e5
    elif n_km > 200.0: penalty += (n_km - 200.0)**2 * 1e5
    
    # Vertical bounds: altitude between 0 and 2500m
    if u_km < min_u_km: penalty += (u_km - min_u_km)**2 * 1e5
    elif u_km > max_u_km: penalty += (u_km - max_u_km)**2 * 1e5
    
    # Exponent bounds (self-calibration constraints)
    # Path loss exponent: typically 1.5 to 2.5
    if eta < 1.5: penalty += (eta - 1.5)**2 * 1e6
    elif eta > 2.5: penalty += (eta - 2.5)**2 * 1e6
    
    # Antenna radiation pattern roll-off: 1.0 to 6.0
    if n_ant < 1.0: penalty += (n_ant - 1.0)**2 * 1e6
    elif n_ant > 6.0: penalty += (n_ant - 6.0)**2 * 1e6
    
    if penalty > 0.0:
        return 1e9 + penalty
        
    rx_ecef = enu_to_ecef_offsets([e_km, n_km, u_km], ref_ecef)
    return compute_objective(rx_ecef, tracks, eta, n_ant, horizon_weight)

def huber_loss(residuals, delta=2.0):
    """Applies Huber loss weighting to handle severe track signal variations."""
    abs_res = np.abs(residuals)
    linear_mask = abs_res > delta
    quadratic_mask = ~linear_mask
    
    out = np.zeros_like(residuals)
    out[quadratic_mask] = 0.5 * (residuals[quadratic_mask] ** 2)
    out[linear_mask] = delta * (abs_res[linear_mask] - 0.5 * delta)
    return np.sum(out)

def compute_objective(rx_ecef, tracks, eta, n_ant, horizon_weight=100.0):
    rx_ecef = np.array(rx_ecef, dtype=float)
    rx_lat, rx_lon, rx_alt = ecef_to_geodetic(rx_ecef[0], rx_ecef[1], rx_ecef[2])
    n_normal = get_local_normal(rx_lat, rx_lon)
    
    total_loss = 0.0
    total_horizon_penalty = 0.0
    total_tracks_processed = 0
    
    for track in tracks:
        pts = track['points']
        n_pts = len(pts)
        if n_pts < 5:
            continue
            
        tx_ecefs = np.array([pt['ecef'] for pt in pts], dtype=float)
        tx_alts = np.array([pt['alt'] for pt in pts], dtype=float)
        rssis = np.array([pt['rssi'] for pt in pts], dtype=float)
        
        r = tx_ecefs - rx_ecef
        dists = np.linalg.norm(r, axis=1)
        dists_clamped = np.maximum(dists, 1.0)
        
        dot_prods = np.sum(r * n_normal, axis=1)
        sin_els = dot_prods / dists_clamped
        sin_els = np.clip(sin_els, -1.0, 1.0)
        cos_els = np.sqrt(1.0 - sin_els**2)
        
        cos_els_clamped = np.maximum(cos_els, 0.08)  
        antenna_gain = 10.0 * n_ant * np.log10(cos_els_clamped)
        
        # Calculate distance-compensated path constant using self-calibrated eta and n_ant
        rssi_corr = rssis - antenna_gain
        c_i = rssi_corr + 10.0 * eta * np.log10(dists_clamped)
        
        c_median = np.median(c_i)
        track_residuals = c_i - c_median
        
        # Factor robust Huber loss into the objective metric
        track_loss = huber_loss(track_residuals, delta=2.0)
        
        # Normalize the loss by the track length to balance different track geometries equally
        total_loss += track_loss / n_pts
        
        # Smooth physical horizon penalty execution (normalized per track)
        h_max_dists = radio_horizon_distance(tx_alts, rx_alt)
        excess = dists - h_max_dists
        excess_penalty = np.sum(np.maximum(0.0, excess)**2)
        total_horizon_penalty += excess_penalty / n_pts
        
        total_tracks_processed += 1
        
    if total_tracks_processed == 0:
        return 1e9
        
    avg_rssi_error = total_loss / total_tracks_processed
    avg_horizon_penalty = total_horizon_penalty / total_tracks_processed
    
    return avg_rssi_error + horizon_weight * (avg_horizon_penalty / 1e6)

def enu_to_ecef_offsets_2d(enu_2d_km, ref_ecef, elevation_spline, spline_lats, spline_lons):
    """Convert local 2D ENU offsets (east, north) in kilometers to ECEF absolute coordinates in meters using local elevation spline."""
    ref_lat, ref_lon, _ = ecef_to_geodetic(ref_ecef[0], ref_ecef[1], ref_ecef[2])
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)
    
    e = enu_2d_km[0] * 1000.0
    n = enu_2d_km[1] * 1000.0
    
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    
    # ECEF translation assuming u = 0 relative to reference coordinate
    dx = -sin_lon * e - sin_lat * cos_lon * n
    dy = cos_lon * e - sin_lat * sin_lon * n
    dz = cos_lat * n
    
    x_zero = ref_ecef[0] + dx
    y_zero = ref_ecef[1] + dy
    z_zero = ref_ecef[2] + dz
    
    lat, lon, _ = ecef_to_geodetic(x_zero, y_zero, z_zero)
    
    # Clamp coordinates to spline grid boundaries to prevent extrapolation errors
    lat_clamped = np.clip(lat, spline_lats[0], spline_lats[-1])
    lon_clamped = np.clip(lon, spline_lons[0], spline_lons[-1])
    
    alt = elevation_spline(lat_clamped, lon_clamped)[0, 0]
    
    # Calculate ECEF with correct altitude
    x, y, z = geodetic_to_ecef(lat, lon, alt)
    return np.array([x, y, z])

def compute_objective_enu_2d(params, ref_ecef, tracks, elevation_spline, spline_lats, spline_lons, horizon_weight=100.0):
    """
    Objective function for 4-parameter fit (e_km, n_km, eta, n_ant).
    Receiver altitude is looked up on the terrain elevation spline.
    """
    e_km, n_km, eta, n_ant = params
    
    penalty = 0.0
    # Horizontal bounds: +/- 200 km
    if e_km < -200.0: penalty += (e_km + 200.0)**2 * 1e5
    elif e_km > 200.0: penalty += (e_km - 200.0)**2 * 1e5
    
    if n_km < -200.0: penalty += (n_km + 200.0)**2 * 1e5
    elif n_km > 200.0: penalty += (n_km - 200.0)**2 * 1e5
    
    # Exponent bounds
    if eta < 1.5: penalty += (eta - 1.5)**2 * 1e6
    elif eta > 2.5: penalty += (eta - 2.5)**2 * 1e6
    
    if n_ant < 1.0: penalty += (n_ant - 1.0)**2 * 1e6
    elif n_ant > 6.0: penalty += (n_ant - 6.0)**2 * 1e6
    
    if penalty > 0.0:
        return 1e9 + penalty
        
    rx_ecef = enu_to_ecef_offsets_2d([e_km, n_km], ref_ecef, elevation_spline, spline_lats, spline_lons)
    return compute_objective(rx_ecef, tracks, eta, n_ant, horizon_weight)

def get_elevation_grid(lat_cpa, lon_cpa):
    """
    Fetch a 5x5 grid of ground elevations (SRTM 90m) centered on the CPA coordinates
    from the public OpenTopoData API. Returns (lats, lons, grid) if successful, else None.
    """
    try:
        # Generate 5x5 coordinate grid in ~30km x 30km box
        lats = np.linspace(lat_cpa - 0.15, lat_cpa + 0.15, 5)
        lons = np.linspace(lon_cpa - 0.15, lon_cpa + 0.15, 5)
        
        loc_strings = []
        for lat in lats:
            for lon in lons:
                loc_strings.append(f"{lat:.5f},{lon:.5f}")
                
        locations = "|".join(loc_strings)
        url = f"https://api.opentopodata.org/v1/srtm90m?locations={locations}"
        
        print(f"[Elevation API] Fetching 5x5 terrain grid from OpenTopoData around seed: Lat={lat_cpa:.4f}, Lon={lon_cpa:.4f}...")
        response = requests.get(url, timeout=3.5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'OK':
                results = data.get('results', [])
                grid = np.zeros((5, 5))
                idx = 0
                for i in range(5):
                    for j in range(5):
                        el = results[idx].get('elevation')
                        if el is None:
                            el = 100.0  # Safe fallback for individual missing grid values
                        grid[i, j] = float(el)
                        idx += 1
                return lats, lons, grid
            else:
                print(f"[Elevation API] OpenTopoData returned status: {data.get('status')}")
        else:
            print(f"[Elevation API] OpenTopoData returned HTTP status: {response.status_code}")
    except Exception as e:
        print(f"[Elevation API] Failed to retrieve terrain elevation: {e}")
    return None

def filter_tracks_dynamically(tracks, max_db_below_peak=12.0, noise_floor_margin=4.0):
    """
    Filters out flat noise floor tails dynamically by setting a threshold above
    the absolute minimum observed RSSI across all tracks. Also clips points
    more than max_db_below_peak below each track's individual peak.
    """
    all_rssis = [pt['rssi'] for t in tracks for pt in t['points']]
    if not all_rssis:
        return tracks
        
    overall_min_rssi = min(all_rssis)
    noise_floor_threshold = overall_min_rssi + noise_floor_margin
    print(f"[Dynamic Filter] Min RSSI across dataset: {overall_min_rssi:.1f}, clipping below: {noise_floor_threshold:.1f}")
    
    filtered = []
    for t in tracks:
        pts = t['points']
        if not pts:
            continue
        max_rssi = max(p['rssi'] for p in pts)
        
        valid_pts = [p for p in pts if p['rssi'] >= max_rssi - max_db_below_peak and p['rssi'] > noise_floor_threshold]
        if len(valid_pts) >= 5:
            filtered.append({
                'icao': t['icao'],
                'points': valid_pts
            })
            
    print(f"[Dynamic Filter] Kept {len(filtered)}/{len(tracks)} tracks after SNR/peak filtering.")
    return filtered

def get_cpa_initial_guess(tracks, assumed_alt=150.0):
    """
    Computes a robust receiver position estimate using the 2D CPA Orthogonality
    Method solved in a local ENU frame. Returns ECEF coordinates.
    """
    cpa_points = []
    for track in tracks:
        pts = track['points']
        if len(pts) < 10:
            continue
            
        times = np.array([pt.get('time', float(i * 10.0)) for i, pt in enumerate(pts)])
        rssis = np.array([pt['rssi'] for pt in pts])
        
        max_idx = np.argmax(rssis)
        start_w = max(0, max_idx - 4)
        end_w = min(len(pts), max_idx + 5)
        w_times = times[start_w:end_w]
        w_rssis = rssis[start_w:end_w]
        
        if len(w_times) < 5:
            continue
            
        try:
            poly_rssi = np.polyfit(w_times, w_rssis, 2)
            a, b, _ = poly_rssi
            if a >= 0:
                t_cpa = times[max_idx]
            else:
                t_cpa = -b / (2.0 * a)
                t_cpa = np.clip(t_cpa, times[0], times[-1])
                
            ecefs = np.array([pt['ecef'] for pt in pts[start_w:end_w]])
            poly_x = np.polyfit(w_times, ecefs[:, 0], 2)
            poly_y = np.polyfit(w_times, ecefs[:, 1], 2)
            poly_z = np.polyfit(w_times, ecefs[:, 2], 2)
            
            x_cpa = np.polyval(poly_x, t_cpa)
            y_cpa = np.polyval(poly_y, t_cpa)
            z_cpa = np.polyval(poly_z, t_cpa)
            p_ecef_cpa = np.array([x_cpa, y_cpa, z_cpa])
            
            v_ecef_cpa = np.array([
                2.0 * poly_x[0] * t_cpa + poly_x[1],
                2.0 * poly_y[0] * t_cpa + poly_y[1],
                2.0 * poly_z[0] * t_cpa + poly_z[1]
            ])
            
            cpa_points.append({
                'p_ecef': p_ecef_cpa,
                'v_ecef': v_ecef_cpa
            })
        except Exception:
            continue
            
    if len(cpa_points) < 2:
        return None
        
    try:
        ref_ecef_temp = np.mean([cp['p_ecef'] for cp in cpa_points], axis=0)
        ref_lat, ref_lon, _ = ecef_to_geodetic(ref_ecef_temp[0], ref_ecef_temp[1], ref_ecef_temp[2])
        ref_ecef = np.array(geodetic_to_ecef(ref_lat, ref_lon, assumed_alt))
        
        A_mat = []
        b_val = []
        
        for cp in cpa_points:
            p_enu = ecef_to_enu_offsets(cp['p_ecef'], ref_ecef) * 1000.0
            p_plus_v_enu = ecef_to_enu_offsets(cp['p_ecef'] + cp['v_ecef'], ref_ecef) * 1000.0
            v_enu = p_plus_v_enu - p_enu
            
            A_mat.append([v_enu[0], v_enu[1]])
            b_val.append(np.dot(v_enu, p_enu))
            
        A_mat = np.array(A_mat)
        b_val = np.array(b_val)
        
        rx_enu_2d, _, _, _ = np.linalg.lstsq(A_mat, b_val, rcond=None)
        rx_enu_m = np.array([rx_enu_2d[0], rx_enu_2d[1], 0.0])
        rx_ecef = enu_to_ecef_offsets(rx_enu_m / 1000.0, ref_ecef)
        return rx_ecef
    except Exception as e:
        print(f"[CPA Guess] Error solving CPA system: {e}")
        return None

def get_initial_guess(tracks):
    """Robust initial guess selector processing top percentiles to ignore packet spikes."""
    all_pts = []
    for track in tracks:
        all_pts.extend(track['points'])
        
    if not all_pts:
        return np.array(geodetic_to_ecef(48.1351, 11.5820, 50.0))
        
    # Extract the 95th percentile highest RSSI point rather than a fragile max()
    rssis = [pt['rssi'] for pt in all_pts]
    threshold = np.percentile(rssis, 95)
    
    candidate_pts = [pt for pt in all_pts if pt['rssi'] >= threshold]
    if candidate_pts:
        mean_lat = np.mean([pt['lat'] for pt in candidate_pts])
        mean_lon = np.mean([pt['lon'] for pt in candidate_pts])
        x, y, z = geodetic_to_ecef(mean_lat, mean_lon, 150.0)
        return np.array([x, y, z])
        
    return np.array(geodetic_to_ecef(48.1351, 11.5820, 50.0))

_elevation_grid_cache = None  # Global cache for elevation grid: (lats, lons, grid)

def get_elevation_grid(lat_cpa, lon_cpa):
    """
    Fetch a 5x5 grid of ground elevations (SRTM 90m) centered on the CPA coordinates
    from the public OpenTopoData API. Returns (lats, lons, grid) if successful, else None.
    Uses an in-memory cache to prevent redundant HTTP requests (e.g. during bootstrap).
    """
    global _elevation_grid_cache
    
    # If cache exists and is close enough (within 0.05 degrees, ~5.5 km), reuse it
    if _elevation_grid_cache is not None:
        cached_lats, cached_lons, cached_grid = _elevation_grid_cache
        cached_lat_center = np.mean(cached_lats)
        cached_lon_center = np.mean(cached_lons)
        if abs(lat_cpa - cached_lat_center) < 0.05 and abs(lon_cpa - cached_lon_center) < 0.05:
            print("[Elevation API] Reusing cached terrain elevation grid.")
            return _elevation_grid_cache
            
    try:
        # Generate 5x5 coordinate grid in ~30km x 30km box
        lats = np.linspace(lat_cpa - 0.15, lat_cpa + 0.15, 5)
        lons = np.linspace(lon_cpa - 0.15, lon_cpa + 0.15, 5)
        
        loc_strings = []
        for lat in lats:
            for lon in lons:
                loc_strings.append(f"{lat:.5f},{lon:.5f}")
                
        locations = "|".join(loc_strings)
        url = f"https://api.opentopodata.org/v1/srtm90m?locations={locations}"
        
        print(f"[Elevation API] Fetching 5x5 terrain grid from OpenTopoData around seed: Lat={lat_cpa:.4f}, Lon={lon_cpa:.4f}...")
        response = requests.get(url, timeout=3.5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'OK':
                results = data.get('results', [])
                grid = np.zeros((5, 5))
                idx = 0
                for i in range(5):
                    for j in range(5):
                        el = results[idx].get('elevation')
                        if el is None:
                            el = 100.0  # Safe fallback for individual missing grid values
                        grid[i, j] = float(el)
                        idx += 1
                
                # Update global cache
                _elevation_grid_cache = (lats, lons, grid)
                return _elevation_grid_cache
            else:
                print(f"[Elevation API] OpenTopoData returned status: {data.get('status')}")
        else:
            print(f"[Elevation API] OpenTopoData returned HTTP status: {response.status_code}")
    except Exception as e:
        print(f"[Elevation API] Failed to retrieve terrain elevation: {e}")
    return None

def estimate_receiver_position(tracks, initial_guess=None, horizon_weight=100.0):
    # 1. Filter tracks dynamically to eliminate noise floor flat tails
    filtered_tracks = filter_tracks_dynamically(tracks)
    
    valid_tracks = [t for t in filtered_tracks if len(t['points']) >= 8]
    if not valid_tracks:
        valid_tracks = [t for t in filtered_tracks if len(t['points']) >= 5]
        if not valid_tracks:
            # Fallback to unfiltered tracks if filtering was too aggressive
            valid_tracks = [t for t in tracks if len(t['points']) >= 5]
            if not valid_tracks:
                raise ValueError("Insufficient tracking database size to compute metrics.")
        
    if initial_guess is None:
        # Try CPA initial guess first
        ref_ecef = get_cpa_initial_guess(valid_tracks)
        if ref_ecef is not None:
            lat_cpa, lon_cpa, alt_cpa = ecef_to_geodetic(ref_ecef[0], ref_ecef[1], ref_ecef[2])
            print(f"[Optimizer] CPA Orthogonality Seed: Lat={lat_cpa:.6f}, Lon={lon_cpa:.6f}, Alt={alt_cpa:.1f}m")
        else:
            ref_ecef = get_initial_guess(valid_tracks)
            lat_init, lon_init, alt_init = ecef_to_geodetic(ref_ecef[0], ref_ecef[1], ref_ecef[2])
            print(f"[Optimizer] CPA seed failed. Using fallback percentile seed: Lat={lat_init:.6f}, Lon={lon_init:.6f}, Alt={alt_init:.1f}m")
    else:
        ref_ecef = np.array(initial_guess)
        
    # Get reference lat/lon for elevation lookup
    ref_lat, ref_lon, _ = ecef_to_geodetic(ref_ecef[0], ref_ecef[1], ref_ecef[2])
    
    # Try fetching elevation grid around reference point
    elevation_grid_data = get_elevation_grid(ref_lat, ref_lon)
    
    if elevation_grid_data is not None:
        lats, lons, grid = elevation_grid_data
        print(f"[Optimizer] Dynamic elevation grid locked: {grid.min():.1f}m to {grid.max():.1f}m. Running 4-parameter terrain-locked optimization...")
        elevation_spline = RectBivariateSpline(lats, lons, grid)
        
        # 4 parameters: e_km, n_km, eta (path loss), n_ant (antenna roll-off)
        u0 = np.array([0.0, 0.0, 2.0, 2.0])
        
        res = minimize(
            compute_objective_enu_2d,
            u0,
            args=(ref_ecef, valid_tracks, elevation_spline, lats, lons, horizon_weight),
            method='Powell',
            options={'maxiter': 1500, 'ftol': 1e-6}
        )
        
        if not res.success:
            res = minimize(
                compute_objective_enu_2d,
                u0,
                args=(ref_ecef, valid_tracks, elevation_spline, lats, lons, horizon_weight),
                method='Nelder-Mead',
                options={'maxiter': 2500, 'xatol': 1e-4, 'fatol': 1e-5}
            )
            
        u_opt = res.x
        x_opt, y_opt, z_opt = enu_to_ecef_offsets_2d(u_opt[:2], ref_ecef, elevation_spline, lats, lons)
        lat, lon, alt = ecef_to_geodetic(x_opt, y_opt, z_opt)
        
        return {
            'ecef': (x_opt, y_opt, z_opt),
            'geodetic': (lat, lon, alt),
            'eta': float(u_opt[2]),
            'n_ant': float(u_opt[3]),
            'enu_offset_km': [u_opt[0], u_opt[1], 0.0],
            'success': res.success,
            'message': res.message,
            'residual': np.sqrt(res.fun),
            'fun': res.fun,
            'terrain_locked': True
        }
    else:
        print("[Optimizer] Elevation API unavailable. Falling back to 5-parameter 3D self-calibration.")
        # 5 parameters: e_km, n_km, u_km, eta (path loss), n_ant (antenna roll-off)
        u0 = np.array([0.0, 0.0, 0.0, 2.0, 2.0])
        
        res = minimize(
            compute_objective_enu,
            u0,
            args=(ref_ecef, valid_tracks, horizon_weight),
            method='Powell',
            options={'maxiter': 1500, 'ftol': 1e-6}
        )
        
        if not res.success:
            res = minimize(
                compute_objective_enu,
                u0,
                args=(ref_ecef, valid_tracks, horizon_weight),
                method='Nelder-Mead',
                options={'maxiter': 2500, 'xatol': 1e-4, 'fatol': 1e-5}
            )
            
        u_opt = res.x
        x_opt, y_opt, z_opt = enu_to_ecef_offsets(u_opt[:3], ref_ecef)
        lat, lon, alt = ecef_to_geodetic(x_opt, y_opt, z_opt)
        
        return {
            'ecef': (x_opt, y_opt, z_opt),
            'geodetic': (lat, lon, alt),
            'eta': float(u_opt[3]),
            'n_ant': float(u_opt[4]),
            'enu_offset_km': u_opt[:3].tolist(),
            'success': res.success,
            'message': res.message,
            'residual': np.sqrt(res.fun),
            'fun': res.fun,
            'terrain_locked': False
        }

def estimate_uncertainty_bootstrap(tracks, est_ecef, n_iterations=15):
    valid_tracks = [t for t in tracks if len(t['points']) >= 5]
    if len(valid_tracks) < 3:
        return {'std_east': 0.0, 'std_north': 0.0, 'std_up': 0.0, 'cov_enu': np.zeros((3, 3))}
        
    estimates_enu = []
    for _ in range(n_iterations):
        sample_indices = np.random.choice(len(valid_tracks), size=len(valid_tracks), replace=True)
        sampled_tracks = [valid_tracks[idx] for idx in sample_indices]
        
        try:
            res = estimate_receiver_position(sampled_tracks, initial_guess=est_ecef)
            if res['success']:
                offset_enu_km = ecef_to_enu_offsets(res['ecef'], est_ecef)
                estimates_enu.append(offset_enu_km * 1000.0)
        except Exception:
            continue
            
    if len(estimates_enu) < 3:
        return {'std_east': 0.0, 'std_north': 0.0, 'std_up': 0.0, 'cov_enu': np.zeros((3, 3))}
        
    estimates_enu = np.array(estimates_enu)
    std_enu = np.std(estimates_enu, axis=0)
    cov_enu = np.cov(estimates_enu, rowvar=False)
    
    return {
        'std_east': float(std_enu[0]),
        'std_north': float(std_enu[1]),
        'std_up': float(std_enu[2]),
        'cov_enu': cov_enu.tolist()
    }