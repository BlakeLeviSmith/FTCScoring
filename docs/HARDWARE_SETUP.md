# Hardware Setup Guide

## Current Hardware: Freenove ESP32-S3 CAM

### Specifications
- **Board**: Freenove ESP32-S3 CAM (16 MB Flash)
- **MCU**: ESP32-S3, dual-core 32-bit 240 MHz
- **Camera**: OV2640 (2 Megapixel)
- **Lens**: 120-degree distortion-free
- **Output**: JPEG via MJPEG stream
- **Connectivity**: Wi-Fi 802.11 b/g/n (AP mode)
- **Resolution**: SVGA (800x600) — configured automatically on startup

### Previous Hardware (Deprecated)
- ESP32-CAM (ESP32-D0WD-V3, 4M Flash + 2M PSRAM) with USB-C TTL programmer
- Had 1-10 second data gaps in the MJPEG stream (hardware limitation)
- Max stable resolution: VGA (640x480)
- Replaced by Freenove ESP32-S3 for better reliability

## Connecting to the Camera

1. Power on the ESP32-S3 board via USB-C
2. The board creates a Wi-Fi access point
3. Connect your computer to the ESP32-S3's WiFi network
4. Verify connection: open `http://192.168.4.1/` in a browser — you should see the camera web interface

### Important Notes
- **Close the browser tab** showing `192.168.4.1` before running the scoring app — the ESP32 can only serve one MJPEG stream client at a time
- The MJPEG stream runs on **port 81**: `http://192.168.4.1:81/stream`
- The web interface (port 80) and the stream (port 81) are separate servers

## Running the Scoring System

```bash
# Basic usage — connects to ESP32 WiFi, configures SVGA, starts dashboard
python app.py

# Open http://localhost:8089 in your browser for the dashboard
```

The app will:
1. Connect to the MJPEG stream at `192.168.4.1:81/stream`
2. Set resolution to SVGA (800x600) and JPEG quality to 15
3. Wait 2 seconds for the camera to stabilize
4. Start the detection and scoring pipeline
5. Print connection health reports every 2 minutes

### Camera Settings
The app automatically configures optimal settings on first connection:
- **Resolution**: SVGA 800x600 (`framesize=11` in Freenove S3 firmware)
- **JPEG Quality**: 15 (lower = better quality, higher = faster)

**Framesize enum differs between firmware versions:**

| Resolution | Freenove S3 (new) | Old ESP32-CAM |
|---|---|---|
| QVGA (320x240) | 6 | 4 |
| VGA (640x480) | 10 | 6 |
| SVGA (800x600) | **11** | 7 |
| XGA (1024x768) | 12 | 8 |

The Freenove S3 also supports a `/resolution` endpoint for fine-grained control:
```
http://192.168.4.1/resolution?sx=1&sy=0&ex=0&ey=0&offx=0&offy=0&tx=800&ty=600&ox=800&oy=600&scale=0&binning=0
```
Where `sx=1` = OV2640 SVGA mode.

## Physical Mounting

### Camera Position
- **Height**: ~5 feet (152 cm) above ground level
- **Distance**: ~5 feet (152 cm) horizontal from the FIELD edge
- **Angle**: Tilted downward to capture both CLASSIFIER/RAMP structures
- **Orientation**: Landscape mode, centered between the two alliance CLASSIFIERs

### Field of View
- 120-degree lens covers approximately 17+ feet horizontal at 5ft distance
- Both alliance RAMPs visible in a single frame
- Balls appear ~14px diameter at VGA, ~17px at SVGA from 65 inches

### Mounting Tips
- Use a stable tripod or clamp-mount
- Ensure camera doesn't vibrate (competition venues are loud)
- Secure cables to prevent accidental disconnection
- Test positioning before matches begin
- Mark the tripod position for consistent placement

## Network Considerations

- ESP32-S3 runs as Wi-Fi AP (access point mode)
- **Only one MJPEG stream client at a time** — close browser tabs showing the camera before running the app
- Wi-Fi range: ~10-15 meters in open space
- Position the processing laptop within range
- The app's connection monitor tracks disconnects and FPS — check terminal output

## Troubleshooting

| Issue | Solution |
|---|---|
| Can't connect to WiFi | Power cycle ESP32-S3 (unplug USB, wait 3s, replug) |
| "Connection aborted" errors | Close browser tabs showing `192.168.4.1`, power cycle ESP32 |
| Stream connects but no video | Check terminal for "First frame" message; if absent, power cycle ESP32 |
| Camera stuck after config change | Power cycle — previous `/control` commands may have corrupted state |
| Low FPS | Check connection monitor report; reduce resolution if needed |
| Colors look wrong | Adjust thresholds in the Detection Tuning panel on the dashboard |
| Image is blurry | Check lens focus ring, clean lens |
| `/capture` returns HTTP 500 | Camera needs power cycle — firmware crashed |

## Discovering Stream URLs

If you're using a different ESP32 board or firmware, use the probe script:

```bash
python probe_stream.py
```

This fetches the camera's HTML page, extracts embedded URLs, and tests common MJPEG endpoints.
