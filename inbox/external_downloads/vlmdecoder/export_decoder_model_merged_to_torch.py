from __future__ import annotations

import shutil
from pathlib import Path

from onnx2torch import convert
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_graph import ValueType

import custom_onnx2torch_ops  # noqa: F401  # Registers custom converters.


def _patch_onnx2torch_node_name_generation() -> None:
    def _generate_node_name(node) -> str:
        safe_domain = node.domain.replace(".", "_")
        raw_name = node.name if node.name else node.op_type
        safe_name = raw_name.replace(".", "_").replace("/", "_")
        return f"{safe_domain}_{safe_name}".strip("_")

    def _value_type(self, value_name: str):
        if value_name == "":
            return ValueType.EMPTY
        if value_name in self._input_values:
            return ValueType.GRAPH_INPUT
        if value_name in self._node_output_values:
            return ValueType.NODE_OUTPUT
        if value_name in self._initializers:
            return ValueType.GRAPH_INITIALIZER
        return ValueType.UNKNOWN

    OnnxGraph.generate_node_name = staticmethod(_generate_node_name)
    OnnxGraph.value_type = _value_type


def main() -> None:
    onnx_path = Path("decoder_model_merged.onnx")
    export_dir = Path("decoder_model_merged_torch")
    top_level_py = Path("decoder_model_merged_torch.py")

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    export_dir.mkdir(parents=True, exist_ok=True)
    _patch_onnx2torch_node_name_generation()

    print(f"[1/3] Convert ONNX -> torch.fx.GraphModule: {onnx_path}")
    model = convert(str(onnx_path))

    print(f"[2/3] Export GraphModule to folder: {export_dir}")
    model.to_folder(export_dir, module_name="DecoderModelMerged")

    generated_module_py = export_dir / "module.py"
    if not generated_module_py.exists():
        raise FileNotFoundError(f"Generated file not found: {generated_module_py}")

    print(f"[3/3] Copy module.py -> {top_level_py}")
    shutil.copy2(generated_module_py, top_level_py)

    print("Done.")
    print(f"Generated folder: {export_dir.resolve()}")
    print(f"Generated top-level file: {top_level_py.resolve()}")


if __name__ == "__main__":
    main()
