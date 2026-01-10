# Vision API Usage Guide

**Endpoint:** `http://192.168.1.10:8088/vision/extract`
**Method:** `POST`
**Format:** `multipart/form-data`

## Expected Input
field name: `file`
content: The image or PDF file you want to analyze.

## Example 1: Using Python
Your friend can run this script to send an image to you:

```python
import requests

# 1. Configuration
HOST_IP = "192.168.1.10"  # Mayur's IP
URL = f"http://{HOST_IP}:8088/vision/extract"
FILE_PATH = "my_image.png" # The file they want to send

# 2. Upload
print(f"Sending {FILE_PATH} to {URL}...")
with open(FILE_PATH, 'rb') as f:
    files = {'file': f}
    response = requests.post(URL, files=files)

# 3. Result
if response.status_code == 200:
    data = response.json()
    print("\n--- Extracted Text ---")
    print(data.get("full_text"))
else:
    print("Error:", response.text)
```

## Example 2: Using cURL (Terminal)
If they use a terminal:
```bash
curl -X POST -F "file=@/path/to/image.png" http://192.168.1.10:8088/vision/extract
```
