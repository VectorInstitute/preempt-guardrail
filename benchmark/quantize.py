"""Write a dynamic INT8 copy of the fp32 ONNX model."""

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_to_int8(source: Path, destination: Path) -> None:
    """Quantize weights to INT8 and leave the fp32 file untouched.

    Dynamic quantization scales activations on each forward pass. It does
    not replace ``model.onnx``. The benchmark promotes INT8 only after
    the accuracy gate passes.

    Parameters
    ----------
    source : Path
        Fp32 ``model.onnx``.
    destination : Path
        Where to write ``model.int8.onnx``. An existing file is overwritten
        so a re-run matches the current fp32 export.

    """
    quantize_dynamic(
        model_input=str(source),
        model_output=str(destination),
        weight_type=QuantType.QInt8,
    )
