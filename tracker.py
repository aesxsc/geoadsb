import time
import requests
import threading
import numpy as np
from coord_math import geodetic_to_ecef

class AircraftTracker:
    def __init__(self, json_url="http://127.0.0.1:8080/data/aircraft.json", poll_interval=1.0):
        self.json_url = json_url
        self.poll_interval = poll_interval
        
        # Track storage: hex -> { 'icao': str, 'points': [ {lat, lon, alt, rssi, time, ecef}, ... ] }
        self.active_tracks = {}
        self.completed_tracks = []
        
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
        
        # Keep track of when we last polled
        self.last_poll_time = 0.0
        self.poll_count = 0
        self.successful_polls = 0
        
        # Stats
        self.total_points_collected = 0

    def start(self):
        with self.lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._poll_loop, daemon=True)
                self.thread.start()
                print(f"Aircraft tracker started. Polling {self.json_url} every {self.poll_interval}s.")

    def stop(self):
        with self.lock:
            self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            print("Aircraft tracker stopped.")

    def _poll_loop(self):
        while True:
            with self.lock:
                if not self.running:
                    break
            
            try:
                self._poll_once()
            except Exception as e:
                print(f"Error during aircraft poll: {e}")
                
            time.sleep(self.poll_interval)

    def _poll_once(self):
        self.poll_count += 1
        now = time.time()
        self.last_poll_time = now
        
        try:
            response = requests.get(self.json_url, timeout=0.8)
            if response.status_code != 200:
                return
            
            data = response.json()
            self.successful_polls += 1
        except Exception:
            # Silently fail for transient network/timeout errors, typical for dump1090 startup
            return

        aircraft_list = data.get('aircraft', [])
        
        with self.lock:
            current_hexes = set()
            for ac in aircraft_list:
                hex_code = ac.get('hex')
                if not hex_code:
                    continue
                
                # Check for valid fields
                lat = ac.get('lat')
                lon = ac.get('lon')
                rssi = ac.get('rssi')
                seen = ac.get('seen', 0)
                
                # We need geometric or barometric altitude
                # dump1090 lists alt_geom or alt_baro or altitude (in feet)
                alt_feet = ac.get('alt_geom')
                if alt_feet is None:
                    alt_feet = ac.get('alt_baro')
                if alt_feet is None:
                    alt_feet = ac.get('altitude')
                    
                # Skip if missing core data or if the report is stale (> 5 seconds ago)
                if lat is None or lon is None or alt_feet is None or rssi is None or seen > 5:
                    continue
                
                # Convert altitude from feet to meters (WGS-84 HAE approximately)
                # Note: alt_geom is geometric height, which is close to Ellipsoidal Height
                # alt_baro is pressure altitude, which has some offset but works fine for localization.
                alt_meters = alt_feet * 0.3048
                
                current_hexes.add(hex_code)
                
                # Parse ECEF coordinates
                x, y, z = geodetic_to_ecef(lat, lon, alt_meters)
                
                pt = {
                    'lat': float(lat),
                    'lon': float(lon),
                    'alt': float(alt_meters),
                    'rssi': float(rssi),
                    'time': now - seen,
                    'ecef': (float(x), float(y), float(z))
                }
                
                if hex_code not in self.active_tracks:
                    self.active_tracks[hex_code] = {
                        'icao': hex_code.strip().upper(),
                        'points': [],
                        'last_seen': now
                    }
                    
                track = self.active_tracks[hex_code]
                
                # Prevent adding duplicate timestamps
                if not track['points'] or abs(track['points'][-1]['time'] - pt['time']) > 0.5:
                    track['points'].append(pt)
                    self.total_points_collected += 1
                
                track['last_seen'] = now

            # Clean up old active tracks that haven't been seen for 60 seconds
            stale_hexes = []
            for h, track in self.active_tracks.items():
                if now - track['last_seen'] > 60.0:
                    stale_hexes.append(h)
                    
            for h in stale_hexes:
                track = self.active_tracks.pop(h)
                # Archive the track if it is of good quality
                if self._is_quality_track(track):
                    self.completed_tracks.append(track)
                    # Limit completed tracks buffer size to prevent memory leaks (keep last 100)
                    if len(self.completed_tracks) > 100:
                        self.completed_tracks.pop(0)

    def _is_quality_track(self, track):
        """
        Check if a track is suitable for optimization.
        Requirements:
        - At least 15 points
        - Covered a distance of at least 10 km
        - Duration of flight is at least 60 seconds
        - Standard deviation of RSSI is at least 2.5 dB (to ensure dynamic range and clear CPA)
        """
        pts = track['points']
        if len(pts) < 15:
            return False
            
        # Time duration
        duration = pts[-1]['time'] - pts[0]['time']
        if duration < 60.0:
            return False
            
        # Geographic distance covered
        p1 = np.array(pts[0]['ecef'])
        p2 = np.array(pts[-1]['ecef'])
        dist = np.linalg.norm(p2 - p1)
        if dist < 10000.0:  # 10 km
            return False
            
        # RSSI variation
        rssis = [pt['rssi'] for pt in pts]
        if np.std(rssis) < 2.5:
            return False
            
        return True

    def get_all_valid_tracks(self):
        """
        Return a list of all tracks (active and completed) that meet quality filters.
        """
        valid_tracks = []
        with self.lock:
            # Check completed tracks
            for t in self.completed_tracks:
                valid_tracks.append(t)
                
            # Check active tracks (copying them)
            for h, t in self.active_tracks.items():
                if self._is_quality_track(t):
                    valid_tracks.append(t)
                    
        return valid_tracks

    def get_state(self):
        """
        Get current tracker state statistics.
        """
        with self.lock:
            n_active = len(self.active_tracks)
            n_completed = len(self.completed_tracks)
            n_valid = len(self.get_all_valid_tracks())
            
            # Count total points in active tracks
            active_points = sum(len(t['points']) for t in self.active_tracks.values())
            completed_points = sum(len(t['points']) for t in self.completed_tracks)
            
            return {
                'active_tracks_count': n_active,
                'completed_tracks_count': n_completed,
                'valid_tracks_count': n_valid,
                'active_points_count': active_points,
                'completed_points_count': completed_points,
                'total_points_collected': self.total_points_collected,
                'poll_count': self.poll_count,
                'successful_polls': self.successful_polls,
                'last_poll_time': self.last_poll_time
            }

    def get_live_aircraft_positions(self):
        """
        Get the most recent position of all active aircraft for dashboard display.
        """
        positions = []
        with self.lock:
            for h, track in self.active_tracks.items():
                if track['points']:
                    last_pt = track['points'][-1]
                    positions.append({
                        'hex': h,
                        'lat': last_pt['lat'],
                        'lon': last_pt['lon'],
                        'alt': last_pt['alt'],
                        'rssi': last_pt['rssi'],
                        'points_count': len(track['points']),
                        'last_seen_sec': time.time() - track['last_seen']
                    })
        return positions
