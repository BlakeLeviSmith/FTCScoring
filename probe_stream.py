"""Quick probe to find the ESP32-S3's actual MJPEG stream URL.
Run while connected to ESP32's WiFi: python probe_stream.py
"""
import requests
import re

base = "http://192.168.4.1"

# Step 1: Fetch the HTML page and look for stream URLs
print("=== Fetching HTML from", base, "===")
try:
    r = requests.get(base, timeout=5)
    html = r.text
    print(f"Status: {r.status_code}, Content-Type: {r.headers.get('Content-Type', '?')}")
    print(f"Length: {len(html)} bytes")
    print()

    # Look for URLs in the HTML
    urls = re.findall(r'(?:src|href|url)\s*[=:(]\s*["\']?([^"\'>\s]+)', html, re.IGNORECASE)
    print("URLs found in HTML:")
    for u in urls:
        print(f"  {u}")
    print()

    # Dump first 3000 chars
    print("=== HTML preview (first 3000 chars) ===")
    print(html[:3000])
except Exception as e:
    print(f"Error: {e}")

print()

# Step 2: Try common stream endpoints
endpoints = [
    ":81/stream", ":81/", "/stream", "/mjpeg", "/video",
    "/capture", "/cam.mjpeg", "/videostream", "/mjpeg/1",
]
print("=== Probing common endpoints ===")
for ep in endpoints:
    url = base + ep if ep.startswith("/") else f"http://192.168.4.1{ep}"
    try:
        r = requests.get(url, stream=True, timeout=3)
        ct = r.headers.get("Content-Type", "?")
        chunk = next(r.iter_content(chunk_size=512), b"")
        has_jpeg = b"\xff\xd8" in chunk
        print(f"  {url:40s} -> {r.status_code} | {ct} | JPEG markers: {has_jpeg}")
        r.close()
    except Exception as e:
        err = str(e)[:80]
        print(f"  {url:40s} -> ERROR: {err}")
