import json
import base64
from io import BytesIO
from PIL import Image
import urllib.request

# Create a small valid 224x224 RGB image
img = Image.new('RGB', (224, 224), color = 'red')
buffered = BytesIO()
img.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

payload = {"image": "base64," + img_str}
with open("test_real.json", "w") as f:
    json.dump(payload, f)

# Try fetching
url = "https://insect-identifier-api.onrender.com/predict"
req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req) as response:
        print("Status", response.status)
        print("Response:", response.read().decode())
except urllib.error.URLError as e:
    print("Failed or crashed:", e)
