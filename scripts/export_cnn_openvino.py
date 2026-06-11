"""
Exporte sleep_model_cnn_best.keras → models/sleep_model_npu.xml + .bin (OpenVINO FP16).

Usage :
  python scripts/export_cnn_openvino.py
  python scripts/export_cnn_openvino.py --model models/sleep_model_cnn_best.keras
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import openvino as ov

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from keras_compat import load_cnn_npu_keras

BATCH_INFER = 64


def main():
    parser = argparse.ArgumentParser(description="Export CNN NPU → OpenVINO IR FP16")
    parser.add_argument(
        "--model",
        default=None,
        help="Chemin .keras (défaut : sleep_model_cnn_best.keras)",
    )
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base, "models")
    model_path = args.model or os.path.join(models_dir, "sleep_model_cnn_best.keras")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    print(f"Chargement : {model_path}")
    model = load_cnn_npu_keras(model_path)
    n_samples = int(model.input_shape[1])

    export_dir = os.path.join(models_dir, "_tmp_saved_model")
    shutil.rmtree(export_dir, ignore_errors=True)

    print(f"Export SavedModel (batch={BATCH_INFER}, shape=[{BATCH_INFER}, {n_samples}, 1])")
    if hasattr(model, "export"):
        model.export(export_dir)
    else:
        import tensorflow as tf

        tf.saved_model.save(model, export_dir)

    ov_model = ov.convert_model(export_dir, input=[(BATCH_INFER, n_samples, 1)])
    ov_model.reshape({ov_model.inputs[0]: [BATCH_INFER, n_samples, 1]})
    xml_path = os.path.join(models_dir, "sleep_model_npu.xml")
    ov.save_model(ov_model, xml_path, compress_to_fp16=True)
    shutil.rmtree(export_dir, ignore_errors=True)

    bin_path = xml_path.replace(".xml", ".bin")
    print(f"OK : {xml_path}")
    if os.path.isfile(bin_path):
        print(f"     {bin_path} ({os.path.getsize(bin_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
