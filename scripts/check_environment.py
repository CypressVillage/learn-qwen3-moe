"""Check the local Python and PyTorch environment without downloading anything."""

import importlib
import sys


def check_python(version_info):
    version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    if (version_info.major, version_info.minor) == (3, 11):
        return {
            "name": "Python",
            "status": "PASS",
            "ok": True,
            "summary": f"Python {version} is supported.",
        }

    return {
        "name": "Python",
        "status": "FAIL",
        "ok": False,
        "summary": (
            f"Python {version} is unsupported. Run `uv python install 3.11.15`, "
            "then `uv sync --locked --python 3.11.15`."
        ),
    }


def check_cpu(torch_module):
    values = torch_module.tensor(
        [1.0, 2.0, 3.0], dtype=torch_module.float32, device="cpu"
    )
    result = (values * values).sum().item()
    return {
        "name": "PyTorch CPU",
        "status": "PASS",
        "ok": True,
        "result": result,
        "summary": f"Deterministic CPU tensor result: {result}.",
    }


def check_cuda(torch_module):
    try:
        if not torch_module.cuda.is_available():
            return {
                "name": "CUDA",
                "status": "SKIPPED",
                "ok": True,
                "available": False,
                "summary": "CUDA is unavailable; CPU-only execution is supported.",
            }

        gpu_name = torch_module.cuda.get_device_name(0)
        total_memory = torch_module.cuda.get_device_properties(0).total_memory
        capability = torch_module.cuda.get_device_capability(0)
        bf16_supported = torch_module.cuda.is_bf16_supported()
        values = torch_module.tensor(
            [1.0, 2.0, 3.0], dtype=torch_module.float32, device="cuda"
        )
        result = (values * values).sum().item()
        torch_version = str(torch_module.__version__)
        cuda_runtime = str(torch_module.version.cuda)
        memory_gib = f"{total_memory / 1024**3:.2f} GiB"
        capability_text = f"{capability[0]}.{capability[1]}"

        return {
            "name": "CUDA",
            "status": "PASS",
            "ok": True,
            "available": True,
            "torch_version": torch_version,
            "cuda_runtime": cuda_runtime,
            "gpu_name": gpu_name,
            "total_memory_bytes": total_memory,
            "total_memory_gib": memory_gib,
            "compute_capability": capability_text,
            "bf16_supported": bf16_supported,
            "result": result,
            "summary": (
                f"PyTorch {torch_version}, CUDA runtime {cuda_runtime}, {gpu_name}, "
                f"{total_memory} bytes ({memory_gib}), capability {capability_text}, "
                f"BF16: {bf16_supported}, tensor result: {result}."
            ),
        }
    except Exception as exc:
        return {
            "name": "CUDA",
            "status": "FAIL",
            "ok": False,
            "available": True,
            "summary": (
                f"{type(exc).__name__}: {exc}. See docs/environment.md for CUDA "
                "troubleshooting guidance."
            ),
        }


def _load_torch():
    return importlib.import_module("torch")


def run_checks(version_info=None, torch_loader=None):
    if version_info is None:
        version_info = sys.version_info
    if torch_loader is None:
        torch_loader = _load_torch

    checks = [check_python(version_info)]
    try:
        torch_module = torch_loader()
    except Exception as exc:
        checks.append(
            {
                "name": "PyTorch",
                "status": "FAIL",
                "ok": False,
                "summary": (
                    f"{type(exc).__name__}: {exc}. Install locked dependencies with "
                    "`uv sync --locked --python 3.11.15`."
                ),
            }
        )
        return {"ok": False, "checks": checks}

    checks.append(
        {
            "name": "PyTorch",
            "status": "PASS",
            "ok": True,
            "summary": f"PyTorch {torch_module.__version__} imported successfully.",
        }
    )
    try:
        checks.append(check_cpu(torch_module))
    except Exception as exc:
        checks.append(
            {
                "name": "PyTorch CPU",
                "status": "FAIL",
                "ok": False,
                "summary": f"{type(exc).__name__}: {exc}.",
            }
        )
    checks.append(check_cuda(torch_module))
    return {"ok": all(check["ok"] for check in checks), "checks": checks}


def main(version_info=None, torch_loader=None, output=print):
    report = run_checks(version_info=version_info, torch_loader=torch_loader)
    for check in report["checks"]:
        output(f'[{check["status"]}] {check["name"]}: {check["summary"]}')

    overall_status = "PASS" if report["ok"] else "FAIL"
    output(f"[{overall_status}] 环境检查 / Environment check")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
