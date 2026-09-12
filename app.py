from flask import Flask, render_template, request, jsonify
import pickle
import requests
import datetime
import numpy as np
import os

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY", "Rl73tmVefZWU7uq35jINfCT3mWPfuryi")
MODEL_PATH = "model.pkl"

# ── Load Model ────────────────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_future_time_features(minutes_ahead=30):
    future = datetime.datetime.now() + datetime.timedelta(minutes=minutes_ahead)
    hour = future.hour
    day = future.weekday() + 1
    is_weekend = int(day >= 6)
    is_rush_hour = int(7 <= hour <= 9 or 17 <= hour <= 19)
    is_night = int(hour < 6 or hour >= 22)
    return hour, day, is_weekend, is_rush_hour, is_night


def get_traffic_data(lat, lon):
    """Fetch real-time traffic speed from TomTom."""
    url = (
        f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        f"?point={lat},{lon}&key={TOMTOM_API_KEY}"
    )
    try:
        r = requests.get(url, timeout=5)
        data = r.json().get("flowSegmentData", {})
        current_speed = data.get("currentSpeed", 50)
        free_speed = data.get("freeFlowSpeed", 60)
        return current_speed, free_speed
    except Exception:
        # Fallback: simulate based on time of day
        hour = datetime.datetime.now().hour
        free_speed = 60
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            current_speed = np.random.uniform(15, 35)
        elif hour < 6 or hour >= 22:
            current_speed = np.random.uniform(50, 65)
        else:
            current_speed = np.random.uniform(35, 55)
        return round(current_speed, 1), free_speed


def predict_congestion(speed, free_speed, hour, day, is_weekend, is_rush_hour, is_night):
    speed_ratio = speed / free_speed if free_speed > 0 else 0.5
    features = [[hour, day, speed, free_speed, speed_ratio, is_weekend, is_rush_hour, is_night]]
    return int(model.predict(features)[0])


def estimate_delay(speed, free_speed, distance_km):
    """Estimate delay in minutes for a segment."""
    if speed <= 0:
        return 0
    normal_time = (distance_km / free_speed) * 60
    actual_time = (distance_km / speed) * 60
    return round(max(0, actual_time - normal_time), 2)


def get_osrm_route(start_lat, start_lon, end_lat, end_lon, alternative=False):
    """Get route(s) from OSRM."""
    alt_param = "true" if alternative else "false"
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}"
        f"?overview=full&geometries=geojson&alternatives={alt_param}&steps=false"
    )
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        routes = []
        for route in data.get("routes", []):
            coords = route["geometry"]["coordinates"]
            # OSRM returns [lon, lat], convert to [lat, lon]
            coords_latlon = [[c[1], c[0]] for c in coords]
            duration_min = route["duration"] / 60
            distance_km = route["distance"] / 1000
            routes.append({
                "coords": coords_latlon,
                "duration_min": round(duration_min, 1),
                "distance_km": round(distance_km, 2)
            })
        return routes
    except Exception as e:
        return []


def sample_route(coords, max_points=20):
    """Sample evenly spaced points along a route."""
    if len(coords) <= max_points:
        return coords
    indices = np.linspace(0, len(coords) - 1, max_points, dtype=int)
    return [coords[i] for i in indices]


def analyze_route(coords, duration_min, distance_km):
    """Analyze congestion along a route and estimate total delay."""
    hour, day, is_weekend, is_rush_hour, is_night = get_future_time_features(30)
    sampled = sample_route(coords, max_points=20)
    segment_distance = distance_km / len(sampled)

    congestion_points = []
    total_delay = 0
    congestion_counts = [0, 0, 0]

    for lat, lon in sampled:
        speed, free_speed = get_traffic_data(lat, lon)
        level = predict_congestion(speed, free_speed, hour, day, is_weekend, is_rush_hour, is_night)
        delay = estimate_delay(speed, free_speed, segment_distance)
        total_delay += delay
        congestion_counts[level] += 1

        congestion_points.append({
            "lat": lat,
            "lon": lon,
            "level": level,
            "speed": speed,
            "free_speed": free_speed,
            "delay_min": delay
        })

    # Dominant congestion level
    dominant = int(np.argmax(congestion_counts))
    estimated_arrival = round(duration_min + total_delay, 1)

    return {
        "points": congestion_points,
        "total_delay_min": round(total_delay, 1),
        "estimated_travel_min": estimated_arrival,
        "dominant_congestion": dominant,
        "congestion_counts": congestion_counts
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    start_lat = float(data["start_lat"])
    start_lon = float(data["start_lon"])
    end_lat = float(data["end_lat"])
    end_lon = float(data["end_lon"])

    # Get primary + alternate routes
    routes = get_osrm_route(start_lat, start_lon, end_lat, end_lon, alternative=True)
    if not routes:
        return jsonify({"error": "Could not calculate route"}), 500

    results = []
    for i, route in enumerate(routes[:2]):  # Max 2 routes
        analysis = analyze_route(route["coords"], route["duration_min"], route["distance_km"])
        results.append({
            "route_index": i,
            "coords": route["coords"],
            "duration_min": route["duration_min"],
            "distance_km": route["distance_km"],
            "estimated_travel_min": analysis["estimated_travel_min"],
            "total_delay_min": analysis["total_delay_min"],
            "dominant_congestion": analysis["dominant_congestion"],
            "congestion_counts": analysis["congestion_counts"],
            "points": analysis["points"]
        })

    # Sort by estimated travel time — best route first
    results.sort(key=lambda r: r["estimated_travel_min"])

    return jsonify({
        "routes": results,
        "prediction_time": "30 min ahead",
        "timestamp": datetime.datetime.now().strftime("%H:%M")
    })


@app.route("/heatmap", methods=["POST"])
def heatmap():
    """Return heatmap data for the bounding box of the route."""
    data = request.json
    points = data.get("points", [])
    hour, day, is_weekend, is_rush_hour, is_night = get_future_time_features(30)

    heatmap_data = []
    for p in points:
        lat, lon = p["lat"], p["lon"]
        speed, free_speed = get_traffic_data(lat, lon)
        level = predict_congestion(speed, free_speed, hour, day, is_weekend, is_rush_hour, is_night)
        intensity = [0.2, 0.6, 1.0][level]
        heatmap_data.append([lat, lon, intensity])

    return jsonify({"heatmap": heatmap_data})


if __name__ == "__main__":
    app.run(debug=True)