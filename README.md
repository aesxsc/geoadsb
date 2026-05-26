# Passive ADS-B Receiver Self-Localization (GeoADSB)

A Python tool and web dashboard to self-localize a single stationary RTL-SDR ADS-B receiver using signal strength (RSSI) telemetry from passing commercial flights. 

By analyzing the signal strength profiles of aircraft as they fly past, the system can triangulate the receiver's geographical coordinates (latitude, longitude, and elevation) to sub-kilometer accuracy. It requires no active GPS module on the receiver and functions entirely passively by listening to standard 1090 MHz Mode S / ADS-B broadcasts.

---

## Installation & Setup

### Prerequisites
1. An RTL-SDR dongle (v3 or v4).
2. A local instance of dump1090. This project is designed to run alongside [gvanem/Dump1090](https://github.com/gvanem/Dump1090) streaming JSON data (typically at `http://127.0.0.1:8080/data/aircraft.json`).
3. Python 3.8 or higher.

### Quick Start
Clone this repository and install the dependencies:
```bash
git clone https://github.com/aesxsc/geoadsb.git
cd geoadsb
pip install -r requirements.txt
```

---

## How to Run

### 1. Live Localization Dashboard
Start your local `dump1090` instance. It is highly recommended to configure it to run at a fixed manual gain (such as `--gain 38.2`) rather than automatic gain control (AGC). AGC dynamically shifts RSSI reference levels, which interferes with absolute path loss modeling.

Start the Flask server:
```bash
python app.py
```

Open **`http://127.0.0.1:5000`** in your browser. 

The background optimizer runs every 15 seconds. As quality flight tracks accumulate, the receiver location marker and its confidence ellipse will update on the map. The dashboard displays the live aircraft trajectories (colored by RSSI), the calculated path loss and antenna constants, and the terrain-locked altitude status.

### 2. Demo / Simulation Mode
If you do not have an RTL-SDR plugged in, you can run the dashboard in simulation mode:
```bash
python app.py --demo
```
This plays back synthetic flights in real time, showing the live map rendering, tracking database compilation, and convergence of the optimizer.

### 3. Verification Test
Run the simulation test suite to verify that the mathematical engine and the dynamic terrain-locking spline work correctly:
```bash
python simulation_test.py
```
This generates synthetic flights around a test receiver coordinate, clamps the signal data to a noisy receiver noise floor, and runs the estimation. The test checks that the horizontal error converges to under 500 meters.

---

## What Makes This Method Unique?

Standard RSSI-based triangulation is notoriously noisy. Multipath fading, building blockage, unknown transponder transmit powers, and receiver automatic gain control (AGC) typically cause signal-strength distance estimates to be off by 10 to 20 kilometers.

To overcome these physical limitations, this project combines several mathematical and geometric techniques:

1. **2D CPA Orthogonality (ENU Frame)**: 
   For each flight track, the system fits a quadratic curve to the RSSI profile to find the exact peak time—the Closest Point of Approach (CPA). It also fits quadratic trajectories to the aircraft's Cartesian coordinates to interpolate its continuous position and velocity vector at the exact moment of CPA. At CPA, the velocity vector and the line-of-sight vector to the receiver are perpendicular. 
   
   Solving this system in standard 3D ECEF coordinates is degenerate because aircraft fly at constant altitudes tangent to the Earth's curvature. We project the positions and velocities into a local East-North-Up (ENU) frame and solve a 2D horizontal system (assuming $u_{rx}=0$). This yields an unbiased starting coordinate seed within 1–3 km of the receiver, even if the antenna is placed next to a window and can only see planes in one direction.

2. **Terrain-Locked Elevation Mapping**: 
   A ground-based receiver is physically bound to the Earth's surface. To prevent the vertical coordinate from drifting to unrealistic heights, the system queries a local 5x5 terrain grid ($30\text{ km} \times 30\text{ km}$) from the public **OpenTopoData (SRTM 90m)** API centered on the initial seed. It builds a continuous 2D bilinear spline (`scipy.interpolate.RectBivariateSpline`) and clamps candidate horizontal coordinates to it. The receiver's altitude is set exactly to the queried ground elevation, reducing the optimization search space from 5 parameters to 4.

3. **Dynamic Noise Floor Filtering**: 
   When aircraft are far away, signal reports hit the hardware noise floor (typically ~-30 dBFS in dump1090) and flatten out. Fitting these flat tails skews the log-distance path loss fit. The system dynamically detects the minimum RSSI across all tracks, clips out points near the noise floor, and keeps only data points within 12 dB of each track's individual peak.

4. **Self-Calibrating Optimization**: 
   The system does not assume a static free-space path loss or antenna radiation pattern. During the non-linear Powell minimization loop, it treats the local terrain path loss exponent ($\eta$) and the antenna vertical gain roll-off ($n_{\text{ant}}$) as free parameters, tuning them on-the-fly to match the physical properties of the receiver's environment.

---

## File Structure

* `tracker.py`: Tracks aircraft by ICAO hex, handles telemetry conversions, and filters out low-quality tracks (requiring at least 15 points, 60s flight duration, 10km distance, and 2.5dB RSSI variation).
* `optimizer.py`: Handles dynamic noise floor filtering, CPA seeding, terrain grid spline mapping, and non-linear Powell/Nelder-Mead optimization.
* `coord_math.py`: Conversions between Geodetic (WGS-84) and Cartesian (ECEF) coordinates, local zenith calculations, and radio horizon limits.
* `app.py`: Flask web server and background optimization worker.
* `templates/index.html`: Dark-themed Leaflet.js dashboard UI.
