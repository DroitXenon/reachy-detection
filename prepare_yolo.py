"""Download/export YOLO26 in an isolated temporary environment (one-time setup)."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("n", "s"), default="s")
    parser.add_argument('--family', choices=('11', '26'), default='26')
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "models"
    root.mkdir(exist_ok=True)
    if args.worker:
        from ultralytics import YOLO

        import onnx

        model = YOLO(str(root / f"yolo{args.family}{args.size}.pt"))
        options = dict(format='onnx', imgsz=640, opset=12, simplify=True,
                       dynamic=False, device='cpu')
        if args.family == '26':
            options.update(nms=False, agnostic_nms=True)
        exported = model.export(**options)
        graph = onnx.load(exported)
        shape = [dimension.dim_value for dimension in graph.graph.output[0].type.tensor_type.shape.dim]
        if args.family == '26' and (len(shape) != 3 or shape[0] != 1 or shape[2] != 6):
            raise RuntimeError(f'Unexpected YOLO26 output shape: {shape}')
        metadata = {"names": model.names, "input_size": 640,
                    "source": f"Ultralytics YOLO{args.family}",
                    "export": "ONNX opset 12, static, float32",
                    "output_format": 'xyxy_score_class' if args.family == '26' else 'xywh_class_scores',
                    "output_shape": shape,
                    "nms_free": args.family == '26',
                    "ultralytics_version": '8.4.193'}
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
                        "--worker", "--size", args.size, '--family', args.family], env=env, check=True)


if __name__ == "__main__":
    main()
