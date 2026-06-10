import asyncio
import json
import base64
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import cv2

from pipeline import SignPredictor, TextBuilder, CameraMetadata

app = FastAPI(title="Bridging Silence")

# --- Model path (reuses the trained model from WINNING) ---
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "WINNING", "model", "mlp_tsl_static.pkl"
)
predictor = SignPredictor(MODEL_PATH)

# --- Static files ---
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    text_builder = TextBuilder()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            # ---------- Frame message ----------
            if msg.get("type") == "frame":
                frame_b64 = msg["frame"]
                if "," in frame_b64:
                    frame_b64 = frame_b64.split(",", 1)[1]

                img_bytes = base64.b64decode(frame_b64)
                arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                cam = msg.get("camera", {})
                meta = CameraMetadata(
                    facing=cam.get("facing", "user"),
                    rotation=cam.get("rotation", 0),
                    mirrored=cam.get("mirrored", False),
                    width=cam.get("width", 640),
                    height=cam.get("height", 480),
                )

                result = predictor.predict_frame(frame, meta)
                state = text_builder.update(result["letter"], result["hand_detected"])

                await ws.send_text(json.dumps({
                    "type": "prediction",
                    "letter": result["letter"],
                    "confidence": result["confidence"],
                    "hand_detected": result["hand_detected"],
                    "word": state["word"],
                    "sentence": state["sentence"],
                    "timestamp": time.time(),
                }))

            # ---------- Command message ----------
            elif msg.get("type") == "command":
                cmd = msg.get("command")
                if cmd == "clear":
                    text_builder.clear()
                elif cmd == "delete_letter":
                    text_builder.delete_letter()
                elif cmd == "delete_word":
                    text_builder.delete_word()
                elif cmd == "speak":
                    await ws.send_text(json.dumps({
                        "type": "speak",
                        "text": text_builder.get_full_text(),
                    }))
                    text_builder.clear()

                await ws.send_text(json.dumps({
                    "type": "state_update",
                    "word": text_builder.word.strip(),
                    "sentence": text_builder.sentence.strip(),
                }))

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WS error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
