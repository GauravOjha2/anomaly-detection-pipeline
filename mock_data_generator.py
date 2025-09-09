#!/usr/bin/env python3
"""
Mock Data Generator for Tourist Safety Anomaly Detection
========================================================

Generates realistic tourist telemetry data with various types of anomalies
for testing the anomaly detection pipeline.

Usage:
    python mock_data_generator.py --scenario normal
    python mock_data_generator.py --scenario emergency
    python mock_data_generator.py --scenario mixed --count 1000
"""

import pandas as pd
import numpy as np
import random
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import json
import os

class TouristDataGenerator:
    def __init__(self, base_lat: float = 39.9042, base_lng: float = 116.4074):
        """Initialize with Beijing coordinates as base location"""
        self.base_lat = base_lat
        self.base_lng = base_lng
        self.tourist_id_counter = 1
        
    def generate_normal_trajectory(self, duration_hours: float = 2.0, points_per_hour: int = 30) -> List[Dict]:
        """Generate normal tourist movement pattern"""
        data = []
        start_time = datetime.now(timezone.utc) - timedelta(hours=duration_hours)
        total_points = int(duration_hours * points_per_hour)
        
        # Normal walking pattern (2-5 km/h)
        base_speed = random.uniform(2.0, 5.0)  # km/h
        current_lat, current_lng = self.base_lat, self.base_lng
        
        for i in range(total_points):
            # More realistic tourist movement (walking around a city)
            lat_delta = random.uniform(-0.002, 0.002)  # ~200m
            lng_delta = random.uniform(-0.002, 0.002)
            
            current_lat += lat_delta
            current_lng += lng_delta
            
            # Normal physiological data
            heart_rate = random.normalvariate(75, 10)
            battery_level = max(20, 100 - (i * 0.5))  # Gradual battery drain
            
            data.append({
                "tourist_id": f"tourist_{self.tourist_id_counter:03d}",
                "lat": round(current_lat, 6),
                "lng": round(current_lng, 6),
                "timestamp": (start_time + timedelta(minutes=i*2)).isoformat(),
                "heart_rate": round(max(40, min(120, heart_rate)), 1),
                "battery_level": round(battery_level, 1),
                "network_status": random.choice(["excellent", "good", "fair"]),
                "panic_button": False,
                "accuracy": round(random.uniform(3.0, 8.0), 1)
            })
        
        self.tourist_id_counter += 1
        return data
    
    def generate_emergency_scenario(self, duration_hours: float = 1.0) -> List[Dict]:
        """Generate emergency scenario with multiple anomaly types"""
        data = []
        start_time = datetime.now(timezone.utc) - timedelta(hours=duration_hours)
        total_points = int(duration_hours * 30)  # 30 points per hour
        
        current_lat, current_lng = self.base_lat, self.base_lng
        
        for i in range(total_points):
            # Emergency movement (running/panic) - much larger movements
            if i < total_points * 0.3:  # First 30% - normal walking
                lat_delta = random.uniform(-0.001, 0.001)  # ~100m
                lng_delta = random.uniform(-0.001, 0.001)
                heart_rate = random.normalvariate(75, 8)
                panic_button = False
            elif i < total_points * 0.7:  # Middle 40% - emergency starts
                lat_delta = random.uniform(-0.01, 0.01)  # Much faster movement ~1km
                lng_delta = random.uniform(-0.01, 0.01)
                heart_rate = random.normalvariate(140, 15)  # High heart rate
                panic_button = i == int(total_points * 0.5)  # Panic button pressed once
            else:  # Last 30% - continued emergency
                lat_delta = random.uniform(-0.005, 0.005)  # Still fast movement
                lng_delta = random.uniform(-0.005, 0.005)
                heart_rate = random.normalvariate(160, 20)  # Very high heart rate
                panic_button = False
            
            current_lat += lat_delta
            current_lng += lng_delta
            
            # Battery drains faster during emergency
            battery_level = max(5, 100 - (i * 1.2))
            
            data.append({
                "tourist_id": f"tourist_{self.tourist_id_counter:03d}",
                "lat": round(current_lat, 6),
                "lng": round(current_lng, 6),
                "timestamp": (start_time + timedelta(minutes=i*2)).isoformat(),
                "heart_rate": round(max(40, min(200, heart_rate)), 1),
                "battery_level": round(battery_level, 1),
                "network_status": random.choice(["poor", "fair"]) if i > total_points * 0.4 else "good",
                "panic_button": panic_button,
                "accuracy": round(random.uniform(5.0, 15.0), 1)  # Lower accuracy during stress
            })
        
        self.tourist_id_counter += 1
        return data
    
    def generate_health_anomaly_scenario(self, duration_hours: float = 1.5) -> List[Dict]:
        """Generate health-related anomaly scenario"""
        data = []
        start_time = datetime.now(timezone.utc) - timedelta(hours=duration_hours)
        total_points = int(duration_hours * 30)
        
        current_lat, current_lng = self.base_lat, self.base_lng
        
        for i in range(total_points):
            # Normal movement
            lat_delta = random.uniform(-0.0008, 0.0008)
            lng_delta = random.uniform(-0.0008, 0.0008)
            current_lat += lat_delta
            current_lng += lng_delta
            
            # Health anomaly patterns
            if i < total_points * 0.2:  # Normal
                heart_rate = random.normalvariate(75, 8)
            elif i < total_points * 0.6:  # Gradual increase
                heart_rate = random.normalvariate(85 + (i * 0.5), 12)
            else:  # Critical levels
                heart_rate = random.normalvariate(180, 25)
            
            # Battery issues
            battery_level = 100 - (i * 0.3)
            if i > total_points * 0.8:  # Critical battery
                battery_level = random.uniform(2, 8)
            
            data.append({
                "tourist_id": f"tourist_{self.tourist_id_counter:03d}",
                "lat": round(current_lat, 6),
                "lng": round(current_lng, 6),
                "timestamp": (start_time + timedelta(minutes=i*2)).isoformat(),
                "heart_rate": round(max(40, min(220, heart_rate)), 1),
                "battery_level": round(max(1, battery_level), 1),
                "network_status": "good" if battery_level > 20 else "poor",
                "panic_button": False,
                "accuracy": round(random.uniform(3.0, 10.0), 1)
            })
        
        self.tourist_id_counter += 1
        return data
    
    def generate_device_failure_scenario(self, duration_hours: float = 1.0) -> List[Dict]:
        """Generate device failure scenario"""
        data = []
        start_time = datetime.now(timezone.utc) - timedelta(hours=duration_hours)
        total_points = int(duration_hours * 30)
        
        current_lat, current_lng = self.base_lat, self.base_lng
        
        for i in range(total_points):
            # Normal movement
            lat_delta = random.uniform(-0.0005, 0.0005)
            lng_delta = random.uniform(-0.0005, 0.0005)
            current_lat += lat_delta
            current_lng += lng_delta
            
            # Device failure patterns
            if i < total_points * 0.3:  # Normal
                battery_level = 100 - (i * 0.2)
                network_status = "excellent"
                accuracy = random.uniform(3.0, 6.0)
            elif i < total_points * 0.7:  # Degrading
                battery_level = max(30, 100 - (i * 0.8))
                network_status = random.choice(["good", "fair"])
                accuracy = random.uniform(6.0, 12.0)
            else:  # Critical failure
                battery_level = random.uniform(1, 15)
                network_status = random.choice(["poor", "no_signal"])
                accuracy = random.uniform(10.0, 25.0)
            
            data.append({
                "tourist_id": f"tourist_{self.tourist_id_counter:03d}",
                "lat": round(current_lat, 6),
                "lng": round(current_lng, 6),
                "timestamp": (start_time + timedelta(minutes=i*2)).isoformat(),
                "heart_rate": random.normalvariate(75, 10) if battery_level > 20 else None,
                "battery_level": round(battery_level, 1),
                "network_status": network_status,
                "panic_button": False,
                "accuracy": round(accuracy, 1)
            })
        
        self.tourist_id_counter += 1
        return data
    
    def generate_extreme_scenario(self, count: int = 50) -> List[Dict]:
        """Generate extreme scenario with obvious anomalies"""
        data = []
        start_time = datetime.now(timezone.utc) - timedelta(hours=1.0)
        
        current_lat, current_lng = self.base_lat, self.base_lng
        
        for i in range(count):
            # Create extreme movements and anomalies
            if i < count * 0.2:  # First 20% - normal
                lat_delta = random.uniform(-0.001, 0.001)
                lng_delta = random.uniform(-0.001, 0.001)
                heart_rate = random.normalvariate(75, 5)
                battery_level = 100 - (i * 0.5)
                panic_button = False
            elif i < count * 0.4:  # Next 20% - extreme movement
                lat_delta = random.uniform(-0.02, 0.02)  # Very large movements
                lng_delta = random.uniform(-0.02, 0.02)
                heart_rate = random.normalvariate(180, 10)  # Very high heart rate
                battery_level = 100 - (i * 1.0)
                panic_button = False
            elif i < count * 0.6:  # Next 20% - panic button
                lat_delta = random.uniform(-0.01, 0.01)
                lng_delta = random.uniform(-0.01, 0.01)
                heart_rate = random.normalvariate(200, 15)  # Extremely high heart rate
                battery_level = 100 - (i * 1.5)
                panic_button = True  # Panic button pressed
            elif i < count * 0.8:  # Next 20% - low battery
                lat_delta = random.uniform(-0.005, 0.005)
                lng_delta = random.uniform(-0.005, 0.005)
                heart_rate = random.normalvariate(160, 20)
                battery_level = random.uniform(5, 15)  # Very low battery
                panic_button = False
            else:  # Last 20% - device failure
                lat_delta = random.uniform(-0.001, 0.001)
                lng_delta = random.uniform(-0.001, 0.001)
                heart_rate = None  # Device failure
                battery_level = random.uniform(1, 5)  # Critical battery
                panic_button = False
            
            current_lat += lat_delta
            current_lng += lng_delta
            
            data.append({
                "tourist_id": f"tourist_{self.tourist_id_counter:03d}",
                "lat": round(current_lat, 6),
                "lng": round(current_lng, 6),
                "timestamp": (start_time + timedelta(minutes=i*2)).isoformat(),
                "heart_rate": round(heart_rate, 1) if heart_rate else None,
                "battery_level": round(battery_level, 1),
                "network_status": "poor" if battery_level < 20 else "good",
                "panic_button": panic_button,
                "accuracy": round(random.uniform(5.0, 25.0), 1)  # Variable accuracy
            })
        
        self.tourist_id_counter += 1
        return data

    def generate_mixed_scenario(self, count: int = 100) -> List[Dict]:
        """Generate mixed scenario with various anomaly types"""
        data = []
        
        # 50% normal data
        normal_count = int(count * 0.5)
        for _ in range(normal_count // 20):  # 20 points per tourist
            data.extend(self.generate_normal_trajectory(duration_hours=0.7))
        
        # 30% extreme scenarios
        extreme_count = int(count * 0.3)
        for _ in range(extreme_count // 25):  # 25 points per extreme scenario
            data.extend(self.generate_extreme_scenario(count=25))
        
        # 20% emergency scenarios
        emergency_count = int(count * 0.2)
        for _ in range(emergency_count // 15):  # 15 points per emergency
            data.extend(self.generate_emergency_scenario(duration_hours=0.5))
        
        # Shuffle the data
        random.shuffle(data)
        return data[:count]  # Return exactly the requested count
    
    def save_to_csv(self, data: List[Dict], filename: str = "mock_tourist_data.csv"):
        """Save data to CSV file"""
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"✅ Saved {len(data)} records to {filename}")
        return filename
    
    def save_to_json(self, data: List[Dict], filename: str = "mock_tourist_data.json"):
        """Save data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {len(data)} records to {filename}")
        return filename

def main():
    parser = argparse.ArgumentParser(description="Generate mock tourist data for anomaly detection")
    parser.add_argument("--scenario", choices=["normal", "emergency", "health", "device", "mixed"], 
                       default="mixed", help="Type of scenario to generate")
    parser.add_argument("--count", type=int, default=200, help="Number of data points to generate")
    parser.add_argument("--output", default="mock_tourist_data", help="Output filename prefix")
    parser.add_argument("--format", choices=["csv", "json", "both"], default="both", help="Output format")
    
    args = parser.parse_args()
    
    print(f"🎭 Generating {args.scenario} scenario with {args.count} data points...")
    
    generator = TouristDataGenerator()
    
    if args.scenario == "normal":
        data = generator.generate_normal_trajectory(duration_hours=args.count/30)
    elif args.scenario == "emergency":
        data = generator.generate_emergency_scenario(duration_hours=args.count/30)
    elif args.scenario == "health":
        data = generator.generate_health_anomaly_scenario(duration_hours=args.count/30)
    elif args.scenario == "device":
        data = generator.generate_device_failure_scenario(duration_hours=args.count/30)
    else:  # mixed
        data = generator.generate_mixed_scenario(count=args.count)
    
    # Save data
    if args.format in ["csv", "both"]:
        generator.save_to_csv(data, f"{args.output}.csv")
    
    if args.format in ["json", "both"]:
        generator.save_to_json(data, f"{args.output}.json")
    
    # Print summary
    print(f"\n📊 Data Summary:")
    print(f"  Total records: {len(data)}")
    print(f"  Unique tourists: {len(set(record['tourist_id'] for record in data))}")
    print(f"  Time range: {data[0]['timestamp']} to {data[-1]['timestamp']}")
    
    # Count anomalies
    panic_count = sum(1 for record in data if record.get('panic_button', False))
    high_hr_count = sum(1 for record in data if record.get('heart_rate', 0) > 150)
    low_battery_count = sum(1 for record in data if record.get('battery_level', 100) < 20)
    
    print(f"\n🚨 Anomaly Summary:")
    print(f"  Panic button presses: {panic_count}")
    print(f"  High heart rate (>150): {high_hr_count}")
    print(f"  Low battery (<20%): {low_battery_count}")

if __name__ == "__main__":
    main()
