from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("models") / "face_landmarker.task"


def main() -> None:
    MODEL_PATH.parent.mkdir(exist_ok=True)
    print("Baixando modelo do rosto...")
    urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
