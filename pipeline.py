import numpy as np
import mediapipe as mp
import joblib
from sklearn.preprocessing import LabelEncoder
import cv2
import time
import base64
from dataclasses import dataclass


@dataclass
class CameraMetadata:
    """Metadata about the camera source for proper frame preprocessing."""
    facing: str = "user"        # "user" (front) or "environment" (back)
    rotation: int = 0           # Device rotation: 0, 90, 180, 270
    mirrored: bool = False      # Whether the browser already mirrored the frame
    width: int = 640
    height: int = 480


class FramePreprocessor:
    """
    Fixes the 3 critical camera variation issues:
    1. Rotation   - phones deliver rotated frames depending on orientation
    2. Mirroring  - front cameras may be pre-mirrored by the browser/OS
    3. Aspect ratio - portrait vs landscape changes landmark distributions
    """

    @staticmethod
    def correct_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
        """Correct frame rotation from device orientation."""
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    @staticmethod
    def correct_mirror(frame: np.ndarray, facing: str, already_mirrored: bool) -> np.ndarray:
        """
        Apply correct mirroring based on camera source.

        Front camera ("user"): we WANT a mirror view for natural UX.
          - If already mirrored by browser -> don't flip
          - If NOT mirrored             -> flip

        Back camera ("environment"): we DON'T want mirror.
          - If already mirrored -> flip to undo
          - If NOT mirrored     -> don't flip
        """
        need_flip = False
        if facing == "user":
            if not already_mirrored:
                need_flip = True
        else:
            if already_mirrored:
                need_flip = True

        return cv2.flip(frame, 1) if need_flip else frame

    @staticmethod
    def normalize_aspect_ratio(frame: np.ndarray, target_ratio: float = 4 / 3) -> np.ndarray:
        """
        Pad frame to match training-data aspect ratio (landscape 4:3).
        This ensures MediaPipe sees the hand in a similar spatial context.
        """
        h, w = frame.shape[:2]
        current_ratio = w / h

        if abs(current_ratio - target_ratio) < 0.05:
            return frame

        if current_ratio < target_ratio:
            # Too tall (portrait) -> pad width
            new_w = int(h * target_ratio)
            pad = (new_w - w) // 2
            frame = cv2.copyMakeBorder(
                frame, 0, 0, pad, new_w - w - pad,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            # Too wide -> pad height
            new_h = int(w / target_ratio)
            pad = (new_h - h) // 2
            frame = cv2.copyMakeBorder(
                frame, pad, new_h - h - pad, 0, 0,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        return frame

    def preprocess(self, frame: np.ndarray, meta: CameraMetadata) -> np.ndarray:
        """Full preprocessing: rotation -> mirror -> aspect ratio."""
        frame = self.correct_rotation(frame, meta.rotation)
        frame = self.correct_mirror(frame, meta.facing, meta.mirrored)
        frame = self.normalize_aspect_ratio(frame)
        return frame


class SignPredictor:
    """Sign language prediction with camera-robust preprocessing."""

    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.le = LabelEncoder()
        self.le.fit([chr(i) for i in range(ord('A'), ord('Z') + 1)])

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.5,
        )
        self.preprocessor = FramePreprocessor()

    @staticmethod
    def normalize_landmarks(landmarks: list) -> np.ndarray:
        """Original per-axis min-max normalization (model-compatible)."""
        coords = np.array(landmarks).reshape(-1, 3).astype(np.float32)
        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        norm = (coords - cmin) / (cmax - cmin + 1e-6)
        return norm.flatten().reshape(1, -1)

    def predict_frame(self, frame: np.ndarray, meta: CameraMetadata = None) -> dict:
        """Process one frame and return prediction."""
        if meta:
            frame = self.preprocessor.preprocess(frame, meta)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        out = {"letter": "", "confidence": 0.0, "hand_detected": False}

        if results.multi_hand_landmarks:
            out["hand_detected"] = True
            lm = results.multi_hand_landmarks[0]
            landmarks = [(p.x, p.y, p.z) for p in lm.landmark]
            try:
                X = self.normalize_landmarks(landmarks)
                idx = self.model.predict(X)[0]
                out["letter"] = self.le.inverse_transform([idx])[0]
                if hasattr(self.model, "predict_proba"):
                    out["confidence"] = float(self.model.predict_proba(X)[0].max())
            except Exception as e:
                out["error"] = str(e)

        return out


class TextBuilder:
    """Builds words and sentences from letter predictions using timing."""

    def __init__(self, hold_s=1.0, space_s=2.0, commit_s=5.0):
        self.hold_s = hold_s
        self.space_s = space_s
        self.commit_s = commit_s
        self.word = ""
        self.sentence = ""
        self.prev_letter = ""
        self.hold_start = None
        self.last_seen = time.time()

    def update(self, letter: str, hand_detected: bool) -> dict:
        now = time.time()

        if hand_detected and letter:
            self.last_seen = now
            if letter == self.prev_letter:
                if self.hold_start is None:
                    self.hold_start = now
                if now - self.hold_start >= self.hold_s:
                    if not self.word or self.word[-1] != letter:
                        self.word += letter
            else:
                self.hold_start = now
            self.prev_letter = letter
        else:
            gap = now - self.last_seen
            if gap >= self.space_s and self.word and not self.word.endswith(" "):
                self.word += " "
            if gap >= self.commit_s and self.word.strip():
                self.sentence += self.word.strip() + " "
                self.word = ""

        return {
            "letter": letter if hand_detected else "",
            "word": self.word.strip(),
            "sentence": self.sentence.strip(),
        }

    def clear(self):
        self.word = ""
        self.sentence = ""
        self.prev_letter = ""
        self.hold_start = None

    def delete_letter(self):
        """
        Deletes the last letter from the active word.
        If active word is empty, pulls the last word from sentence and deletes its last letter.
        Resets hold timers so the currently held gesture is not immediately re-added on the next frame.
        """
        if self.word.strip():
            self.word = self.word.rstrip()[:-1]
        elif self.sentence.strip():
            words = self.sentence.strip().split()
            last_word = words[-1]
            remaining = words[:-1]
            self.sentence = " ".join(remaining) + (" " if remaining else "")
            self.word = last_word[:-1]

        # Reset hold timer & previous letter tracking to prevent auto-re-adding
        self.hold_start = time.time()
        self.prev_letter = ""

    def delete_word(self):
        if self.word.strip():
            self.word = ""
        elif self.sentence.strip():
            words = self.sentence.strip().split()
            self.sentence = " ".join(words[:-1]) + " " if len(words) > 1 else ""

    def get_full_text(self) -> str:
        return (self.sentence + self.word).strip()
