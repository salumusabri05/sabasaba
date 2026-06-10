# Bridging Silence (TSL) WebSocket API Integration Guide

This guide explains how to connect and stream video frames to the **Bridging Silence** real-time sign language recognition server.

---

## 1. Connection Endpoints

* **Production (Railway):** `wss://your-railway-app-url.up.railway.app/ws`
* **Local Development:** `ws://127.0.0.1:8000/ws`

---

## 2. Communication Flow

The protocol is fully bidirectional over a single persistent WebSocket connection:
1. **Client** captures camera frames (Base64 JPEG) and streams them alongside metadata.
2. **Server** processes frames using MediaPipe and the trained ML model, returning predictions.
3. **Client** can send commands (e.g., clear, speak) to manage text accumulation.

```
Sequence Flow:
[Mobile/Web App]                       [FastAPI WebSocket Server]
       |                                           |
       |-------- Connect (wss://...) ------------->|
       |<------- Connection established -----------|
       |                                           |
       |-- Send Frame (Base64 + Camera metadata) ->|
       |<------- Return Prediction (Confidence) ---| (Repeats ~10x per sec)
       |                                           |
       |-------- Send Command ("clear") ---------->|
       |<------- Return updated sentence ----------|
```

---

## 3. Data Payloads

### A. Sending a Video Frame (Client → Server)
Send this JSON payload every **100ms (10 frames per second)**.
Avoid sending at standard 30 FPS to prevent network congestion.

```json
{
  "type": "frame",
  "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "camera": {
    "facing": "user",
    "rotation": 270,
    "mirrored": true,
    "width": 640,
    "height": 480
  }
}
```

#### Field Specifications:
* `frame`: The JPEG image formatted as a standard data URL (`data:image/jpeg;base64,...`).
* `camera.facing`: `"user"` (front camera) or `"environment"` (back camera).
* `camera.rotation`: The raw device/camera orientation angle in degrees (`0`, `90`, `180`, or `270`).
* `camera.mirrored`: Set to `true` if the camera preview is mirrored (common on front cameras).
* `camera.width` & `camera.height`: The dimensions of the captured frame.

---

### B. Sending a Control Command (Client → Server)
Send these commands to control the server's internal text builder:

```json
{
  "type": "command",
  "command": "clear"
}
```
* **Supported Commands:**
  * `"clear"`: Resets the entire accumulated text sentence.
  * `"delete"`: Deletes the last character in the active sentence.
  * `"speak"`: Tells the server to prepare or trigger TTS.

---

### C. Receiving Predictions (Server → Client)
The server responds to every frame and command with this format:

```json
{
  "status": "success",
  "letter": "A",
  "confidence": 0.94,
  "accumulated_text": "HELLO WORLD",
  "hand_detected": true
}
```

#### Field Specifications:
* `letter`: The currently predicted gesture/letter (returns `null` if no hand is detected).
* `confidence`: The model's confidence value between `0.0` and `1.0`.
* `accumulated_text`: The current sentence compiled so far.
* `hand_detected`: Boolean indicating if MediaPipe has successfully tracked a hand in the frame.

---

## 4. Mobile Performance Guidelines

1. **Resolution:** Scale down camera captures to **640x480** or **480x360** before encoding. Higher resolutions increase payload size without improving model accuracy.
2. **Compression:** Set JPEG quality compression to **60% - 70%**. A frame size of **15KB - 30KB** is ideal.
3. **Throttling:** Do not stream raw frames directly from the camera callback. Use a timer/interval to send a frame once every **100ms** to guarantee low latency.
