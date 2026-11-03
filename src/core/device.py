"""Probe the inference backend without downloading a model."""

import ctypes
import sys
from pydantic import Field
from src.model.options import ValidatedModel


class DeviceRequest(ValidatedModel):
    device_index: int = Field(default=0, ge=0)


class DeviceReport(ValidatedModel):
    cuda_available: bool = False
    device_count: int = 0
    device_index: int = 0
    compute_types: list[str] = Field(default_factory=list)
    cpu_compute_types: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    note: str = "Preflight only; model loading and inference may still fail due to memory or runtime errors."


def check_device(device_index: int = 0) -> DeviceReport:
    """Validate a GPU index and report CUDA capabilities and missing runtime libraries."""
    request = DeviceRequest(device_index=device_index)
    report = DeviceReport(device_index=request.device_index)
    try:
        # Check devices and compute types via CTranslate2, which is used by faster-whisper.
        import ctranslate2

        report.cpu_compute_types = sorted(ctranslate2.get_supported_compute_types("cpu"))
        report.device_count = ctranslate2.get_cuda_device_count()
        if report.device_count <= request.device_index:
            report.issues.append(
                f"CUDA device {request.device_index} is unavailable ({report.device_count} visible devices). Check the NVIDIA driver and CUDA_VISIBLE_DEVICES."
            )
            return report
        report.compute_types = sorted(ctranslate2.get_supported_compute_types("cuda", request.device_index))

        if not report.compute_types:
            report.issues.append("CTranslate2 reports no supported CUDA compute types.")

        # CTranslate2 can enumerate GPUs even when Whisper runtime libraries are absent.
        libraries = (
            ["cublas64_12.dll", "cudnn64_9.dll"] if sys.platform == "win32" else ["libcublas.so.12", "libcudnn.so.9"]
        )
        for library in libraries:
            try:
                ctypes.CDLL(library)
            except OSError as exc:
                report.issues.append(
                    f"Cannot load {library}: {exc}. Configure the CUDA 12/cuDNN 9 library search path before starting Python."
                )
        report.cuda_available = bool(report.compute_types) and not report.issues
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        report.issues.append(f"CTranslate2 device probe failed: {exc}")
    return report


def select_device(device, compute_type, device_index=0):
    """Choose a usable device and compute type, rejecting unsupported explicit choices."""
    report = check_device(device_index)
    selected = device.value
    if selected == "auto":
        selected = "cuda" if report.cuda_available else "cpu"
    if selected == "cuda" and not report.cuda_available:
        raise RuntimeError("CUDA is unavailable: " + "; ".join(report.issues))
    supported = report.compute_types if selected == "cuda" else report.cpu_compute_types
    if not supported:
        raise RuntimeError("No supported inference backend: " + "; ".join(report.issues))
    if compute_type == "auto":
        preferences = ["float16", "int8_float16", "float32"] if selected == "cuda" else ["int8", "float32"]
        compute_type = next((kind for kind in preferences if kind in supported), supported[0])
    if compute_type not in supported:
        raise ValueError(f'{compute_type} is unsupported on {selected}; supported: {", ".join(supported)}')
    return selected, compute_type
