# seed_data.py — run once to add demo data
from database import save_detection
from datetime import datetime, timedelta
import random

locations = [
    {"lat": 23.5204, "lng": 87.3119},
    {"lat": 23.5350, "lng": 87.2950},
    {"lat": 23.4990, "lng": 87.3280},
]

severities = [
    (0, []),
    (1, [[100,200,180,280]]),
    (3, [[100,200,180,280],[300,150,400,250],[500,300,600,400]]),
    (7, [[50,100,200,200],[250,100,400,200],[450,100,600,200],
         [50,250,200,350],[250,250,400,350],[450,250,600,350],[150,180,350,330]]),
]

for i in range(20):
    count, boxes = random.choice(severities)
    confs        = [round(random.uniform(0.55, 0.95), 2) for _ in boxes]
    loc          = random.choice(locations)
    source       = random.choice(["image","video","webcam"])

    save_detection(
        image_path       = f"results/demo_{i}.jpg",
        potholes_found   = count > 0,
        boxes            = boxes,
        confidence_scores= confs,
        source           = source,
        location         = loc,
    )

print("✅ 20 sample detections added to MongoDB!")
print("Refresh your dashboard to see the data.")
