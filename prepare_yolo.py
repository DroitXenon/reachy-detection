"""Download/export YOLO11 in an isolated temporary environment (one-time setup)."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("n", "s"), default="s")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "models"
    root.mkdir(exist_ok=True)
    if args.worker:
        from ultralytics import YOLO

        model = YOLO(str(root / f"yolo11{args.size}.pt"))
        exported = model.export(format="onnx", imgsz=640, opset=12, simplify=True,
                                dynamic=False, device="cpu")
        metadata = {"names": model.names, "input_size": 640,
                    "source": "Ultralytics YOLO11",
                    "export": "ONNX opset 12, static, no embedded NMS"}
        Path(exported).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Ready: {exported}")
        return
    # Ultralytics requires opencv-python while MediaPipe requires contrib. Keep
    # export dependencies isolated so the project's cv2 install is not overwritten.
    with tempfile.TemporaryDirectory(prefix="reachy-yolo-export-") as temporary:
        environment = Path(temporary) / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(executable), "-m", "pip", "install", "ultralytics==8.4.193",
                        "onnx==1.23.2", "onnxslim==0.1.98", "onnxruntime==1.31.1"], check=True)
        env = dict(os.environ, YOLO_CONFIG_DIR=str(Path(temporary) / "settings"))
        subprocess.run([str(executable), str(Path(__file__).resolve()),
                        "--worker", "--size", args.size], env=env, check=True)


if __name__ == "__main__":
    main()
