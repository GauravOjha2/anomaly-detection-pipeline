import requests

url = "http://127.0.0.1:8000/detect_batch"

trajectory = [
    {"tourist_id": "u001", "lat": 39.984702, "lng": 116.318417, "timestamp": "2008-10-23T05:54:00Z"},
    {"tourist_id": "u001", "lat": 39.984683, "lng": 116.318450, "timestamp": "2008-10-23T05:55:00Z"},
    {"tourist_id": "u001", "lat": 39.984686, "lng": 116.318417, "timestamp": "2008-10-23T05:56:00Z"},
    {"tourist_id": "u001", "lat": 39.984688, "lng": 116.318385, "timestamp": "2008-10-23T05:57:00Z"},
    {"tourist_id": "u001", "lat": 39.985000, "lng": 116.319000, "timestamp": "2008-10-23T05:58:00Z"}  # jump
]

trajectory[-1]["panic_button"] = True
# or:
trajectory[-1]["timestamp"] = "2008-10-23T05:54:01Z"
trajectory[-1]["lat"] = 40.5
trajectory[-1]["lng"] = 117.5

response = requests.post(url, json=trajectory)
print("Status code:", response.status_code)
print("Response:", response.json())

