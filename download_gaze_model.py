from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

# Peso oficial do repositorio yakhyo/gaze-estimation (MIT), hospedado no GitHub Releases
# (link estavel, diferente do Google Drive do L2CS-Net original que ficou fora do ar).
MODEL_URL = "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx"
MODEL_PATH = Path(__file__).with_name("models") / "mobileone_s0_gaze.onnx"


def main() -> None:
    MODEL_PATH.parent.mkdir(exist_ok=True)
    print("Baixando modelo de gaze (MobileGaze, ~5MB)...")
    urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
