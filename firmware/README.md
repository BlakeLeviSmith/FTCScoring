# firmware/

**Third-party.** Stock Freenove ESP32-S3 CAM firmware (CameraWebServer).

Included only so the camera-side setup is reproducible. This isn't our
code — we don't modify the firmware. We talk to it over the WiFi AP at
`http://192.168.4.1/stream` (MJPEG on port 81). See
`docs/HARDWARE_SETUP.md` for the camera setup.
