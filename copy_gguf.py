#!/usr/bin/env python3
"""
Minimal script to copy a GGUF file to a new GGUF file, preserving all tensors
and metadata.

Based on the pattern in gguf_new_metadata.py but with no argument parsing or
metadata override logic.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure gguf-py is importable
_script_dir = Path(__file__).parent
if (Path.cwd() / "gguf-py").exists():
    sys.path.insert(0, str(Path.cwd() / "gguf-py"))
elif (_script_dir / "gguf-py").exists():
    sys.path.insert(0, str(_script_dir / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter  # noqa: E402

# ---------------------------------------------------------------------------
# Hardcoded input / output paths — change these as needed
# ---------------------------------------------------------------------------
INPUT_PATH = "/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/blobs/ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"
MTP_PATH = "/root/.cache/huggingface/hub/models--am17an--Qwen3.6-35BA3B-MTP-GGUF/blobs/b6d91455942ed408b45318af7312a9fe02a6e48564bf303735770e3fd3523d4f"
OUTPUT_PATH = "/root/.cache/huggingface/hub/Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf"

MTP_BLOCK_INDEX = 40

def main() -> None:
    # 1. Open the source GGUF file for reading
    reader = GGUFReader(INPUT_PATH, "r")
    mtp_reader = GGUFReader(MTP_PATH, "r")

    # 2. Extract architecture and endianness from the source
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()

    # 3. Create the output writer
    writer = GGUFWriter(OUTPUT_PATH, arch=arch, endianess=reader.endianess)

    # Preserve source alignment if set
    alignment_field = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment_field is not None:
        writer.data_alignment = alignment_field.contents()

    # 4. Copy all KV metadata fields (skip virtual fields)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    # 5. Register all tensors with the writer
    for tensor in reader.tensors:
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )


    ##########

    for field_key in [f"{arch}.block_count", f"{arch}.nextn_predict_layers"]:
        field = mtp_reader.get_field(field_key)
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    

    mtp_tensors = [t for t in mtp_reader.tensors if f"blk.{MTP_BLOCK_INDEX}." in t.name]

    print(f"Found {len(mtp_tensors)} MTP tensors:")
    for tensor in mtp_tensors:
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )

    ##########


    # 6. Write header, KV data, and tensor info
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # 7. Write tensor data
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)

    for tensor in mtp_tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=mtp_reader.endianess)

    # 8. Clean up
    writer.close()


if __name__ == "__main__":
    main()

