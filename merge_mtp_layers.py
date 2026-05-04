#!/usr/bin/env python3
"""
Merge MTP (Multi-Token Prediction) layers from a source GGUF into a base GGUF.

This script copies the entire blk.{MTP_BLOCK_INDEX} decoder block from a source
GGUF (typically a higher-precision model with MTP enabled) into a base GGUF
(typically a quantized model without MTP), producing a merged output GGUF.

The MTP block is treated as an additional decoder block by the llama.cpp architecture.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure gguf-py is importable
_script_dir = Path(__file__).parent
if (Path.cwd() / "gguf-py").exists():
    sys.path.insert(0, str(Path.cwd() / "gguf-py"))
elif (_script_dir / "gguf-py").exists():
    sys.path.insert(0, str(_script_dir / "gguf-py"))

import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# User-configurable constants
# ==============================================================================

# Path to the base GGUF (quantized model without MTP layers)
BASE_GGUF_PATH = "/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/blobs/ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"

# Path to the source GGUF (with MTP layers, e.g., Q8_0)
MTP_SOURCE_GGUF_PATH = "/root/.cache/huggingface/hub/models--am17an--Qwen3.6-35BA3B-MTP-GGUF/blobs/b6d91455942ed408b45318af7312a9fe02a6e48564bf303735770e3fd3523d4f"

# Output path for the merged GGUF
OUTPUT_GGUF_PATH = "/root/.cache/huggingface/hub/Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf"

# The block index of the MTP block in the source GGUF.
# The source has 41 blocks (0..40); blk.40 is the MTP block.
MTP_BLOCK_INDEX = 40

# ==============================================================================
# Helper functions
# ==============================================================================


def find_mtp_tensors(reader: GGUFReader, block_index: int) -> list[gguf.ReaderTensor]:
    """Find all tensors belonging to the MTP block (blk.{block_index}) in the given reader."""
    prefix = f"blk.{block_index}."
    return [t for t in reader.tensors if prefix in t.name]


def get_architecture(reader: GGUFReader) -> str:
    """Extract the architecture name from GGUF metadata."""
    arch_field = reader.fields["general.architecture"]
    arch = arch_field.contents(0)
    if isinstance(arch, bytes):
        arch = arch.decode("utf-8")
    return arch


def format_size(nbytes: int) -> str:
    """Format byte count to human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} PB"


# ==============================================================================
# Main logic
# ==============================================================================


def main() -> None:
    base_path = Path(BASE_GGUF_PATH)
    source_path = Path(MTP_SOURCE_GGUF_PATH)
    output_path = Path(OUTPUT_GGUF_PATH)

    # Validate input files
    if not base_path.exists():
        raise FileNotFoundError(f"Base GGUF not found: {BASE_GGUF_PATH}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source GGUF not found: {MTP_SOURCE_GGUF_PATH}")

    # -------------------------------------------------------------------------
    # Step 1: Read both GGUF files
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1: Reading GGUF files")
    logger.info("=" * 60)

    logger.info(f"Reading base GGUF: {BASE_GGUF_PATH}")
    base_reader = GGUFReader(BASE_GGUF_PATH)
    base_arch = get_architecture(base_reader)
    logger.info(f"  Architecture: {base_arch}")
    logger.info(f"  Total tensors: {len(base_reader.tensors)}")
    logger.info(f"  Total size: {format_size(sum(t.n_bytes for t in base_reader.tensors))}")

    logger.info(f"Reading source GGUF: {MTP_SOURCE_GGUF_PATH}")
    source_reader = GGUFReader(MTP_SOURCE_GGUF_PATH)
    source_arch = get_architecture(source_reader)
    logger.info(f"  Architecture: {source_arch}")
    logger.info(f"  Total tensors: {len(source_reader.tensors)}")
    logger.info(f"  Total size: {format_size(sum(t.n_bytes for t in source_reader.tensors))}")

    arch = base_arch

    # -------------------------------------------------------------------------
    # Step 2: Find MTP tensors in the source
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Step 2: Finding MTP tensors (blk.{MTP_BLOCK_INDEX})")
    logger.info("=" * 60)

    mtp_tensors = find_mtp_tensors(source_reader, MTP_BLOCK_INDEX)
    if len(mtp_tensors) == 0:
        raise ValueError(f"No tensors found for blk.{MTP_BLOCK_INDEX} in source GGUF")

    logger.info(f"Found {len(mtp_tensors)} MTP tensors:")
    for t in mtp_tensors:
        qtype_str = t.tensor_type.name if t.tensor_type else "unknown"
        shape_str = "x".join(str(d) for d in t.shape)
        logger.info(f"  [{qtype_str:>6s}] {t.name} ({shape_str}) [{format_size(t.n_bytes)}]")

    # -------------------------------------------------------------------------
    # Step 3: Read metadata from source
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 3: Reading metadata from source GGUF")
    logger.info("=" * 60)

    block_count_key = f"{arch}.block_count"
    nextn_key = f"{arch}.nextn_predict_layers"

    source_block_count = int(source_reader.fields[block_count_key].contents(0))
    logger.info(f"  Source {block_count_key} = {source_block_count}")

    source_nextn_predict_layers = int(source_reader.fields[nextn_key].contents(0))
    logger.info(f"  Source {nextn_key} = {source_nextn_predict_layers}")

    # -------------------------------------------------------------------------
    # Step 4: Create output GGUF
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 4: Writing output GGUF")
    logger.info("=" * 60)

    # Determine use_temp_file based on output size
    total_estimated_size = sum(t.n_bytes for t in base_reader.tensors) + sum(
        t.n_bytes for t in mtp_tensors
    )
    use_temp_file = total_estimated_size > 2 * 1024 * 1024 * 1024  # 2GB

    logger.info(f"  Output path: {OUTPUT_GGUF_PATH}")
    logger.info(f"  Estimated size: {format_size(total_estimated_size)}")
    logger.info(f"  Using temp file: {use_temp_file}")

    # Create GGUFWriter
    writer = GGUFWriter(
        path=OUTPUT_GGUF_PATH,
        arch=arch,
        use_temp_file=use_temp_file,
    )

    # -------------------------------------------------------------------------
    # Step 5: Copy metadata from base (skip keys that will be overridden)
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 5a: Copying metadata from base GGUF")

    skip_keys = {
        block_count_key,
        nextn_key,
        "GGUF.tensor_count",
        "GGUF.kv_count",
    }

    for key, field in base_reader.fields.items():
        if key in skip_keys:
            continue
        val = field.contents()
        vtype = field.types[0]
        writer.add_key_value(key, val, vtype)
        logger.info(f"  [META] {key} = ")

    # -------------------------------------------------------------------------
    # Step 6: Override metadata with source values
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 5b: Overriding metadata from source GGUF")

    writer.add_key_value(block_count_key, source_block_count, gguf.GGUFValueType.UINT32)
    logger.info(f"  Setting {block_count_key} = {source_block_count}")

    writer.add_key_value(nextn_key, source_nextn_predict_layers, gguf.GGUFValueType.UINT32)
    logger.info(f"  Setting {nextn_key} = {source_nextn_predict_layers}")

    # -------------------------------------------------------------------------
    # Step 7: Copy tensors from base
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 6: Copying tensors from base GGUF")

    mtp_prefix = f"blk.{MTP_BLOCK_INDEX}."
    base_tensors_copied = 0
    base_bytes_copied = 0

    for tensor in base_reader.tensors:
        # Skip tensors belonging to the MTP block
        if mtp_prefix in tensor.name:
            logger.info(f"  [SKIP] {tensor.name} (MTP block in base)")
            continue

        writer.add_tensor(tensor.name, tensor.data)
        base_tensors_copied += 1
        base_bytes_copied += tensor.n_bytes

    logger.info(f"  Copied {base_tensors_copied} tensors from base ({format_size(base_bytes_copied)})")

    # -------------------------------------------------------------------------
    # Step 8: Copy MTP tensors from source
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 7: Copying MTP tensors from source GGUF")

    mtp_tensors_copied = 0
    mtp_bytes_copied = 0

    for tensor in mtp_tensors:
        writer.add_tensor(tensor.name, tensor.data)
        mtp_tensors_copied += 1
        mtp_bytes_copied += tensor.n_bytes

    logger.info(f"  Copied {mtp_tensors_copied} MTP tensors ({format_size(mtp_bytes_copied)})")

    # -------------------------------------------------------------------------
    # Step 9: Finalize
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 8: Finalizing output GGUF")
    logger.info("=" * 60)

    total_tensors = base_tensors_copied + mtp_tensors_copied
    total_bytes = base_bytes_copied + mtp_bytes_copied

    logger.info(f"  Total tensors: {total_tensors}")
    logger.info(f"  Total size: {format_size(total_bytes)}")

    # Write everything
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUCCESS: Merged GGUF written to")
    logger.info(f"  {OUTPUT_GGUF_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
