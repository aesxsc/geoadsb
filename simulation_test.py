import numpy as np
from coord_math import geodetic_to_ecef, ecef_to_geodetic, calculate_elevation_angle, radio_horizon_distance
from optimizer import estimate_receiver_position, estimate_uncertainty_bootstrap

def generate_simulated_data(true_lat, true_lon, true_alt, num_tracks=10, noise_std=1.0):
    """
    Generate synthetic ADS-B tracks relative to a true receiver position.
    """
    true_ecef = np.array(geodetic_to_ecef(true_lat, true_lon, true_alt))
    tracks = []
    
    print(f"Generating simulation data with true position: Lat={true_lat:.6f}, Lon={true_lon:.6f}, Alt={true_alt:.1f}m")
    
    np.random.seed(42)  # For reproducible simulation tests
    
    for t_idx in range(num_tracks):
        # 1. Choose track-specific constants
        # Aircraft transmit power constant C_j (randomized between -15 and -5 dBFS)
        c_j = np.random.uniform(-15.0, -5.0)
        
        # 2. Define a straight-line flight path
        # Random starting angle from receiver
        bearing = np.random.uniform(0, 2 * np.pi)
        # CPA distance (closest point of approach) between 5 km and 80 km
        cpa_dist = np.random.uniform(5000.0, 80000.0)
        # Flight path angle relative to receiver radial direction (around perpendicular for good CPA)
        flight_angle = bearing + np.pi/2 + np.random.uniform(-np.pi/6, np.pi/6)
        
        # Flight velocity: 150 to 280 m/s (~300 to 550 knots)
        speed = np.random.uniform(150.0, 280.0)
        # Constant flight altitude: 3,000m to 12,000m
        altitude = np.random.uniform(3000.0, 12000.0)
        
        # Closest approach coordinate in geodetic relative terms (approximate meters offset)
        # Convert true position to local ENU offsets
        cpa_e = cpa_dist * np.sin(bearing)
        cpa_n = cpa_dist * np.cos(bearing)
        
        # Run track for 30 steps (e.g. every 10 seconds for 300 seconds)
        points = []
        for step in range(30):
            t = (step - 15) * 10  # from -150s to +150s around CPA
            
            # Position in local ENU relative to receiver
            e = cpa_e + speed * t * np.sin(flight_angle)
            n = cpa_n + speed * t * np.cos(flight_angle)
            
            # Simple local flat-earth approximation for generating path points
            # 1 deg lat ~ 111,000m, 1 deg lon ~ 111,000m * cos(lat)
            lat_offset = n / 111320.0
            lon_offset = e / (111320.0 * np.cos(np.radians(true_lat)))
            
            pt_lat = true_lat + lat_offset
            pt_lon = true_lon + lon_offset
            
            pt_ecef = np.array(geodetic_to_ecef(pt_lat, pt_lon, altitude))
            
            # Compute true distance
            dist = np.linalg.norm(pt_ecef - true_ecef)
            
            # Compute true elevation angle
            el = calculate_elevation_angle(true_ecef, pt_ecef, rx_lat_lon=(true_lat, true_lon))
            
            # Verify if aircraft is above the radio horizon
            max_range = radio_horizon_distance(altitude, true_alt)
            if dist > max_range:
                # Below horizon, skip this point
                continue
                
            # Compute antenna pattern correction (G = 20 * log10(cos(el)))
            cos_el = np.cos(np.radians(el))
            cos_el = np.maximum(cos_el, 0.05)  # clamp to 87 degrees
            antenna_gain = 20.0 * np.log10(cos_el)
            
            # Compute path loss: S_i = C_j - 20 * log10(d_i) + G_i
            # Adding noise
            noise = np.random.normal(0.0, noise_std)
            rssi = c_j - 20.0 * np.log10(dist) + antenna_gain + noise
            
            points.append({
                'lat': pt_lat,
                'lon': pt_lon,
                'alt': altitude,
                'ecef': pt_ecef.tolist(),
                'rssi': rssi
            })
            
        if len(points) >= 10:
            tracks.append({
                'icao': f'sim_{t_idx:02d}',
                'points': points
            })
            
    return tracks

def run_simulation_test():
    # True receiver position (Munich, Germany)
    true_lat = 48.1351
    true_lon = 11.5820
    true_alt = 520.0  # 520 meters elevation
    
    true_ecef = np.array(geodetic_to_ecef(true_lat, true_lon, true_alt))
    
    # Generate 50 tracks with 1.0 dB RSSI noise
    tracks = generate_simulated_data(true_lat, true_lon, true_alt, num_tracks=50, noise_std=1.0)
    print(f"Generated {len(tracks)} valid tracks for simulation.")
    
    # Select the top 10 tracks with the highest maximum RSSI (closest passes, highest SNR)
    tracks.sort(key=lambda t: max(pt['rssi'] for pt in t['points'] if 'rssi' in pt) if t['points'] else -999, reverse=True)
    tracks = tracks[:10]
    print(f"Selected top {len(tracks)} tracks with highest RSSI for optimization.")
    
    # Generate an initial guess perturbed by ~15 km horizontally and 200m vertically
    init_lat = true_lat + 0.12  # approx 13.3 km north
    init_lon = true_lon - 0.08  # approx 6.0 km west
    init_alt = true_alt + 200.0  # 200m higher
    init_ecef = geodetic_to_ecef(init_lat, init_lon, init_alt)
    
    print("\nStarting Position Estimation...")
    res = estimate_receiver_position(tracks, initial_guess=init_ecef)
    
    est_lat, est_lon, est_alt = res['geodetic']
    est_ecef = res['ecef']
    
    print("\n--- ESTIMATION RESULTS ---")
    print(f"Success: {res['success']} ({res['message']})")
    print(f"True Position: Lat={true_lat:.6f}, Lon={true_lon:.6f}, Alt={true_alt:.1f}m")
    print(f"Est. Position: Lat={est_lat:.6f}, Lon={est_lon:.6f}, Alt={est_alt:.1f}m")
    
    # Calculate errors in meters
    error_ecef = np.array(est_ecef) - true_ecef
    # Local ENU conversion to see horizontal and vertical errors
    lat_rad = np.radians(true_lat)
    lon_rad = np.radians(true_lon)
    R = np.array([
        [-np.sin(lon_rad), np.cos(lon_rad), 0.0],
        [-np.sin(lat_rad)*np.cos(lon_rad), -np.sin(lat_rad)*np.sin(lon_rad), np.cos(lat_rad)],
        [np.cos(lat_rad)*np.cos(lon_rad), np.cos(lat_rad)*np.sin(lon_rad), np.sin(lat_rad)]
    ])
    error_enu = np.dot(R, error_ecef)
    horizontal_error = np.sqrt(error_enu[0]**2 + error_enu[1]**2)
    vertical_error = np.abs(error_enu[2])
    
    print(f"\nHorizontal Position Error: {horizontal_error:.2f} meters")
    print(f"Vertical Elevation Error:   {vertical_error:.2f} meters")
    print(f"Total 3D Coordinate Error:  {np.linalg.norm(error_ecef):.2f} meters")
    print(f"Est. Path Loss Exponent (eta):  {res['eta']:.3f} (True: 2.000)")
    print(f"Est. Antenna Roll-Off (n_ant): {res['n_ant']:.3f} (True: 2.000)")
    print(f"Average RSSI Residual:      {res['residual']:.3f} dB")
    
    print("\nEstimating Uncertainty via Bootstrap...")
    unc = estimate_uncertainty_bootstrap(tracks, est_ecef, n_iterations=10)
    print(f"Estimated Standard Deviations (Confidence bounds):")
    print(f"  East:  ±{unc['std_east']:.2f} meters")
    print(f"  North: ±{unc['std_north']:.2f} meters")
    print(f"  Up:    ±{unc['std_up']:.2f} meters")
    
    # Accept test if 3D error is less than 500 meters (given 1.0 dB RSSI noise, this is extremely precise!)
    assert np.linalg.norm(error_ecef) < 500.0, "Test failed: Estimation error too high!"
    print("\n[SUCCESS] Mathematical model validation passed successfully!")

if __name__ == '__main__':
    run_simulation_test()
