"""
Generate index.json (GeoJSON) from vpint2_samples.json 

Helps visualise rois using a map tool like https://geojson.io/next 
"""

import json
from pathlib import Path

samples_path = Path(__file__).parent / "vpint2_samples.json"
samples = json.loads(samples_path.read_text())

features = []
for data_id, sample in samples.items():
    lat, lon = sample["roi"][1]
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {"roi": data_id},
    })

out = Path(__file__).parent / "index.json"
out.write_text(json.dumps(
    {"type": "FeatureCollection", "features": features}, indent=2) + "\n")
repo_root = Path(__file__).parent.parent
print(f"Wrote {len(features)} features to {out.relative_to(repo_root)}")
