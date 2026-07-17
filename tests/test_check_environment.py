import importlib
import importlib.util
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "check_environment.py"
SPEC = importlib.util.spec_from_file_location("check_environment", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_environment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_environment)
check_cpu = check_environment.check_cpu
check_cuda = check_environment.check_cuda
check_python = check_environment.check_python
main = check_environment.main
run_checks = check_environment.run_checks


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def __mul__(self, other):
        return FakeTensor([left * right for left, right in zip(self.values, other.values)])

    def sum(self):
        return FakeTensor([sum(self.values)])

    def item(self):
        return self.values[0]


class FakeCuda:
    def __init__(self, available=False, failure=None):
        self.available = available
        self.failure = failure

    def is_available(self):
        return self.available

    def get_device_name(self, index):
        if self.failure == "query":
            raise RuntimeError("driver query exploded")
        return "NVIDIA Test GPU"

    def get_device_properties(self, index):
        return SimpleNamespace(total_memory=8 * 1024**3)

    def get_device_capability(self, index):
        return (8, 9)

    def is_bf16_supported(self):
        return True


class FakeTorch:
    float32 = "float32"
    __version__ = "2.7.1+cu128"
    version = SimpleNamespace(cuda="12.8")

    def __init__(self, cuda, tensor_failure_device=None):
        self.cuda = cuda
        self.tensor_failure_device = tensor_failure_device
        self.tensor_calls = []

    def tensor(self, values, *, dtype, device):
        if device == self.tensor_failure_device:
            raise RuntimeError(f"{device} tensor operation exploded")
        self.tensor_calls.append((values, dtype, device))
        return FakeTensor(values)


def test_python_311_is_accepted():
    report = check_python(SimpleNamespace(major=3, minor=11, micro=9))

    assert report["ok"] is True
    assert report["status"] == "PASS"
    assert "3.11.9" in report["summary"]


@pytest.mark.parametrize("major, minor", [(3, 10), (3, 12), (4, 0)])
def test_other_python_major_minor_is_rejected_with_uv_guidance(major, minor):
    report = check_python(SimpleNamespace(major=major, minor=minor, micro=1))

    assert report["ok"] is False
    assert report["status"] == "FAIL"
    assert "uv python install 3.11.15" in report["summary"]
    assert "uv sync --locked --python 3.11.15" in report["summary"]


def test_cpu_check_runs_a_deterministic_tensor_operation():
    torch = FakeTorch(cuda=FakeCuda())

    report = check_cpu(torch)

    assert report["ok"] is True
    assert report["status"] == "PASS"
    assert report["result"] == 14.0
    assert torch.tensor_calls == [([1.0, 2.0, 3.0], "float32", "cpu")]


def test_cuda_unavailable_is_skipped_and_not_a_failure():
    torch = FakeTorch(cuda=FakeCuda())

    report = check_cuda(torch)

    assert report["ok"] is True
    assert report["status"] == "SKIPPED"
    assert report["available"] is False
    assert "CUDA" in report["summary"]


def test_importing_script_has_no_output_or_torch_loading(capsys, monkeypatch):
    def unexpected_import(name):
        raise AssertionError(f"unexpected deferred import: {name}")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)

    runpy.run_path(str(SCRIPT_PATH), run_name="import_safety_check")

    assert capsys.readouterr() == ("", "")


def test_missing_torch_fails_with_sync_guidance_and_exception_summary():
    def missing_torch():
        raise ModuleNotFoundError("No module named 'torch'")

    report = run_checks(
        version_info=SimpleNamespace(major=3, minor=11, micro=15),
        torch_loader=missing_torch,
    )

    assert report["ok"] is False
    torch_report = report["checks"][1]
    assert torch_report["status"] == "FAIL"
    assert "ModuleNotFoundError: No module named 'torch'" in torch_report["summary"]
    assert "uv sync --locked --python 3.11.15" in torch_report["summary"]


def test_no_gpu_environment_succeeds_when_python_torch_and_cpu_pass():
    torch = FakeTorch(cuda=FakeCuda())

    report = run_checks(
        version_info=SimpleNamespace(major=3, minor=11, micro=15),
        torch_loader=lambda: torch,
    )

    assert report["ok"] is True
    assert [check["status"] for check in report["checks"]] == [
        "PASS",
        "PASS",
        "PASS",
        "SKIPPED",
    ]


def test_cuda_available_reports_hardware_and_runs_deterministic_operation():
    torch = FakeTorch(cuda=FakeCuda(available=True))

    report = check_cuda(torch)

    assert report == {
        "name": "CUDA",
        "status": "PASS",
        "ok": True,
        "available": True,
        "torch_version": "2.7.1+cu128",
        "cuda_runtime": "12.8",
        "gpu_name": "NVIDIA Test GPU",
        "total_memory_bytes": 8 * 1024**3,
        "total_memory_gib": "8.00 GiB",
        "compute_capability": "8.9",
        "bf16_supported": True,
        "result": 14.0,
        "summary": (
            "PyTorch 2.7.1+cu128, CUDA runtime 12.8, NVIDIA Test GPU, "
            "8589934592 bytes (8.00 GiB), capability 8.9, BF16: True, "
            "tensor result: 14.0."
        ),
    }
    assert torch.tensor_calls == [([1.0, 2.0, 3.0], "float32", "cuda")]


@pytest.mark.parametrize(
    "torch, exception_summary",
    [
        (FakeTorch(cuda=FakeCuda(available=True, failure="query")), "driver query exploded"),
        (
            FakeTorch(cuda=FakeCuda(available=True), tensor_failure_device="cuda"),
            "cuda tensor operation exploded",
        ),
    ],
)
def test_cuda_failure_fails_overall_with_exception_and_docs_guidance(
    torch, exception_summary
):
    report = run_checks(
        version_info=SimpleNamespace(major=3, minor=11, micro=15),
        torch_loader=lambda: torch,
    )

    assert report["ok"] is False
    cuda_report = report["checks"][-1]
    assert cuda_report["status"] == "FAIL"
    assert f"RuntimeError: {exception_summary}" in cuda_report["summary"]
    assert "docs/environment.md" in cuda_report["summary"]


def test_cli_uses_stable_labels_and_returns_zero_on_success():
    lines = []

    exit_code = main(
        version_info=SimpleNamespace(major=3, minor=11, micro=15),
        torch_loader=lambda: FakeTorch(cuda=FakeCuda()),
        output=lines.append,
    )

    assert exit_code == 0
    assert lines[0].startswith("[PASS] Python:")
    assert lines[-1] == "[PASS] 环境检查 / Environment check"


def test_cli_returns_one_and_prints_failure():
    lines = []

    exit_code = main(
        version_info=SimpleNamespace(major=3, minor=12, micro=1),
        torch_loader=lambda: FakeTorch(cuda=FakeCuda()),
        output=lines.append,
    )

    assert exit_code == 1
    assert lines[0].startswith("[FAIL] Python:")
    assert lines[-1] == "[FAIL] 环境检查 / Environment check"
