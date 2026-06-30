"""
Model export helpers.
- Save Keras model to SavedModel format
- (Optional) export to ONNX if tf2onnx is installed
- For TensorRT, export ONNX then run:
    trtexec --onnx=model.onnx --saveEngine=model.plan --explicitBatch --fp16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description="Export Keras model.")
    parser.add_argument("--model", required=True, help="Input Keras model path (.keras or SavedModel)")
    parser.add_argument("--savedmodel-out", default="savedmodel", help="Directory to save SavedModel")
    parser.add_argument("--onnx-out", help="Path to save ONNX (requires tf2onnx)")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    tf.saved_model.save(model, args.savedmodel_out)
    print(f"SavedModel exported to {args.savedmodel_out}")

    if args.onnx_out:
        try:
            import tf2onnx  # type: ignore
        except ImportError:
            print("tf2onnx not installed; skip ONNX export.")
            return
        spec = (tf.TensorSpec((None, model.input_shape[1], model.input_shape[2]), tf.float32, name="input"),)
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        with open(args.onnx_out, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX exported to {args.onnx_out}")


if __name__ == "__main__":
    main()
