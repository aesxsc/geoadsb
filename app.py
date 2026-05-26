import os
import time
import argparse
import threading
from flask import Flask, render_template, jsonify, request
import numpy as np

from coord_math import geodetic_to_ecef, ecef_to_geodetic, calculate_elevation_angle, radio_horizon_distance
from optimizer import estimate_receiver_position, estimate_uncertainty_bootstrap
from tracker import AircraftTracker
from simulation_test import generate_simulated_data

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Global variables for optimization results
estimated_position = None
uncertainty_stats = None
is_optimizing = False
demo_mode = False
tracker = None

# Custom configuration variables
dump1090_url = "http://127.0.0.1:8080/data/aircraft.json"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    state = tracker.get_state()
    
    # Add optimization results
    opt_data = None
    if estimated_position is not None:
        lat, lon, alt = estimated_position['geodetic']
        opt_data = {
            'lat': lat,
            'lon': lon,
            'alt': alt,
            'eta': estimated_position.get('eta', 2.0),
            'n_ant': estimated_position.get('n_ant', 2.0),
            'residual': estimated_position['residual'],
            'success': estimated_position['success'],
            'message': estimated_position['message'],
            'enu_offset_km': estimated_position['enu_offset_km'],
            'terrain_locked': estimated_position.get('terrain_locked', False)
        }
        
        if uncertainty_stats is not None:
            opt_data['uncertainty'] = {
                'std_east': uncertainty_stats['std_east'],
                'std_north': uncertainty_stats['std_north'],
                'std_up': uncertainty_stats['std_up']
            }
            
    return jsonify({
        'tracker_state': state,
        'estimated_position': opt_data,
        'is_optimizing': is_optimizing,
        'demo_mode': demo_mode,
        'dump1090_url': dump1090_url
    })

@app.route('/api/aircraft')
def get_aircraft():
    return jsonify(tracker.get_live_aircraft_positions())

@app.route('/api/tracks')
def get_tracks():
    valid_tracks = tracker.get_all_valid_tracks()
    
    # Format tracks for JSON transmission (only sending essential fields to save bandwidth)
    formatted = []
    for t in valid_tracks:
        pts = []
        for pt in t['points']:
            pts.append({
                'lat': pt['lat'],
                'lon': pt['lon'],
                'alt': pt['alt'],
                'rssi': pt['rssi']
            })
        formatted.append({
            'icao': t['icao'],
            'points': pts
        })
    return jsonify(formatted)

@app.route('/api/recalculate', methods=['POST'])
def force_recalculate():
    global estimated_position, uncertainty_stats, is_optimizing
    if is_optimizing:
        return jsonify({'status': 'error', 'message': 'Optimization already in progress'}), 409
        
    valid_tracks = tracker.get_all_valid_tracks()
    if len(valid_tracks) < 3:
        return jsonify({'status': 'error', 'message': f'Insufficient quality tracks. Need at least 3, have {len(valid_tracks)}.'}), 400
        
    # Run in a separate thread to avoid blocking the HTTP request
    def run_opt():
        global estimated_position, uncertainty_stats, is_optimizing
        is_optimizing = True
        try:
            res = estimate_receiver_position(valid_tracks)
            estimated_position = res
            unc = estimate_uncertainty_bootstrap(valid_tracks, res['ecef'])
            uncertainty_stats = unc
        except Exception as e:
            print(f"Error in manual recalculation: {e}")
        finally:
            is_optimizing = False
            
    threading.Thread(target=run_opt, daemon=True).start()
    return jsonify({'status': 'ok', 'message': 'Optimization triggered'})


def background_optimizer():
    """
    Background worker that runs the optimization every 30 seconds
    if new quality tracks have been added.
    """
    global estimated_position, uncertainty_stats, is_optimizing
    last_valid_count = 0
    
    while True:
        time.sleep(15.0)  # Check every 15 seconds
        valid_tracks = tracker.get_all_valid_tracks()
        n_valid = len(valid_tracks)
        
        # We need at least 3 tracks to estimate 3D position
        if n_valid >= 3 and (n_valid > last_valid_count or estimated_position is None) and not is_optimizing:
            is_optimizing = True
            try:
                print(f"[Optimizer Thread] Starting self-localization using {n_valid} flight tracks...")
                # Run optimization (can use initial guess if available)
                init_guess = estimated_position['ecef'] if estimated_position else None
                res = estimate_receiver_position(valid_tracks, initial_guess=init_guess)
                estimated_position = res
                
                # Bootstrap uncertainty (std deviations)
                unc = estimate_uncertainty_bootstrap(valid_tracks, res['ecef'])
                uncertainty_stats = unc
                
                last_valid_count = n_valid
                print(f"[Optimizer Thread] Complete: Lat={res['geodetic'][0]:.6f}, Lon={res['geodetic'][1]:.6f}, Alt={res['geodetic'][2]:.1f}m, Residual={res['residual']:.3f} dB")
            except Exception as e:
                print(f"[Optimizer Thread] Error during optimization: {e}")
            finally:
                is_optimizing = False


def demo_feed_worker(true_lat=48.1351, true_lon=11.5820, true_alt=520.0):
    """
    Simulates live aircraft transmissions by feeding simulated flights into the tracker.
    Runs only if --demo mode is selected.
    """
    print(f"[Demo Mode] Simulating aircraft around True Position: Lat={true_lat}, Lon={true_lon}, Alt={true_alt}m")
    
    # Generate static database of 30 mock tracks
    sim_tracks = generate_simulated_data(true_lat, true_lon, true_alt, num_tracks=30, noise_std=1.0)
    
    # Play them back in real-time
    # We will feed points from each track sequentially
    pt_indices = {i: 0 for i in range(len(sim_tracks))}
    
    while True:
        time.sleep(1.0)  # Feed points every 1 second
        
        with tracker.lock:
            now = time.time()
            tracker.last_poll_time = now
            tracker.poll_count += 1
            tracker.successful_polls += 1
            
            # Select 3-6 planes that are currently "flying"
            active_sim_indices = [i for i in range(len(sim_tracks)) if pt_indices[i] < len(sim_tracks[i]['points'])]
            
            # If all flights finished, reset them to run again (continuous demo)
            if not active_sim_indices:
                print("[Demo Mode] All mock flights completed. Resetting tracks for loop...")
                pt_indices = {i: 0 for i in range(len(sim_tracks))}
                # Clear tracker so it restarts estimation
                tracker.active_tracks.clear()
                tracker.completed_tracks.clear()
                global estimated_position, uncertainty_stats
                estimated_position = None
                uncertainty_stats = None
                continue
            
            # Select a random subset to update this tick (to simulate packet loss/varying reports)
            active_subset = np.random.choice(active_sim_indices, size=min(len(active_sim_indices), 8), replace=False)
            
            for idx in active_subset:
                track = sim_tracks[idx]
                pt_idx = pt_indices[idx]
                pt_data = track['points'][pt_idx]
                
                hex_code = track['icao']
                
                # Convert feet altitude back for tracker input simulation
                alt_feet = pt_data['alt'] / 0.3048
                
                # Build mock aircraft json structure
                pt = {
                    'lat': pt_data['lat'],
                    'lon': pt_data['lon'],
                    'alt': pt_data['alt'],
                    'rssi': pt_data['rssi'],
                    'time': now,
                    'ecef': pt_data['ecef']
                }
                
                if hex_code not in tracker.active_tracks:
                    tracker.active_tracks[hex_code] = {
                        'icao': hex_code,
                        'points': [],
                        'last_seen': now
                    }
                    
                ac_track = tracker.active_tracks[hex_code]
                ac_track['points'].append(pt)
                ac_track['last_seen'] = now
                tracker.total_points_collected += 1
                
                pt_indices[idx] += 1
                
            # Process timeouts and archiving
            # In demo mode, if an index is completed, archive it immediately
            for i in range(len(sim_tracks)):
                hex_code = sim_tracks[i]['icao']
                if hex_code in tracker.active_tracks and pt_indices[i] >= len(sim_tracks[i]['points']):
                    ac_track = tracker.active_tracks.pop(hex_code)
                    if tracker._is_quality_track(ac_track):
                        tracker.completed_tracks.append(ac_track)
                        if len(tracker.completed_tracks) > 100:
                            tracker.completed_tracks.pop(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GeoADSB SDR Self-Localization Server")
    parser.add_argument('--url', type=str, default="http://127.0.0.1:8080/data/aircraft.json",
                        help="dump1090 aircraft.json URL")
    parser.add_argument('--port', type=int, default=5000, help="Port to run the Flask web server")
    parser.add_argument('--demo', action='store_true', help="Run in Demo Mode with simulated flights")
    parser.add_argument('--demo-lat', type=float, default=48.1351, help="Demo True Latitude (Munich)")
    parser.add_argument('--demo-lon', type=float, default=11.5820, help="Demo True Longitude (Munich)")
    parser.add_argument('--demo-alt', type=float, default=520.0, help="Demo True Altitude in meters (Munich)")
    
    args = parser.parse_args()
    
    dump1090_url = args.url
    demo_mode = args.demo
    
    # Initialize Aircraft Tracker
    tracker = AircraftTracker(json_url=dump1090_url, poll_interval=1.0)
    
    if demo_mode:
        # Run demo feed in background thread
        threading.Thread(
            target=demo_feed_worker, 
            args=(args.demo_lat, args.demo_lon, args.demo_alt),
            daemon=True
        ).start()
    else:
        # Start regular dump1090 poller
        tracker.start()
        
    # Start background optimizer thread
    threading.Thread(target=background_optimizer, daemon=True).start()
    
    # Launch Web Server
    print(f"Launching web interface at http://127.0.0.1:{args.port}/")
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
