"""Read CPU tensors through the DLPack protocol (``__dlpack__``).

DLPack is how PyTorch, JAX, NumPy and others exchange tensors, including
formats NumPy cannot describe (bfloat16, float8). The C reads the capsule's
structures (DLPack 1.3's layouts, ``csrc/module.c``) and copies the elements.
"""

from __future__ import annotations

import sys

from . import _native, formats
from .array import Array, row_major_strides
from .formats import UnsupportedFormatError

# DLDataTypeCode values from dlpack.h.
_CODES = {
    (0, 8): formats.int8, (0, 16): formats.int16, (0, 32): formats.int32, (0, 64): formats.int64,
    (1, 8): formats.uint8, (1, 16): formats.uint16, (1, 32): formats.uint32, (1, 64): formats.uint64,
    (2, 16): formats.float16, (2, 32): formats.float32, (2, 64): formats.float64,
    (4, 16): formats.bfloat16,
    (5, 32): formats.complex32, (5, 64): formats.complex64, (5, 128): formats.complex128,
    (6, 8): formats.bool_,
    (7, 8): formats.float8_e3m4,
    (8, 8): formats.float8_e4m3,
    (9, 8): formats.float8_e4m3b11fnuz,
    (10, 8): formats.float8_e4m3fn,
    (11, 8): formats.float8_e4m3fnuz,
    (12, 8): formats.float8_e5m2,
    (13, 8): formats.float8_e5m2fnuz,
}
_CPU_DEVICES = (1, 3, 11)  # kDLCPU, kDLCUDAHost, kDLROCMHost: memory the CPU can read.
_MAX_VERSION = (1, 3)

def from_dlpack(obj) -> Array:
    """Copy a CPU tensor exposing ``__dlpack__`` into an ``Array``."""
    device_type, device_id = obj.__dlpack_device__()
    if int(device_type) not in _CPU_DEVICES:
        raise ValueError(
            f"DLPack device type {int(device_type)} (id {device_id}) is not CPU memory; copy the tensor to the CPU first"
        )
    try:
        capsule = obj.__dlpack__(max_version=_MAX_VERSION)
    except TypeError:
        capsule = obj.__dlpack__()
    owner, (major, code, bits, lanes, shape, strides, address) = _native.dlpack_open(capsule)
    try:
        if major is not None and major != _MAX_VERSION[0]:
            raise UnsupportedFormatError(f"unsupported DLPack major version {major}")
        dtype = _CODES.get((code, bits))
        if dtype is None or lanes != 1:
            raise UnsupportedFormatError(f"unsupported DLPack dtype (code={code}, bits={bits}, lanes={lanes})")
        data = _native.gather(address, shape, row_major_strides(shape) if strides is None else strides, dtype.itemsize)
        return Array(dtype, shape, data, byteorder=sys.byteorder)
    finally:
        _native.dlpack_close(owner)
