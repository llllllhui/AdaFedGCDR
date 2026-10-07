"""设备解析与浮点后端开关。

所有子命令通过 :func:`resolve_device` 接受 ``--device``：``auto`` / ``cpu`` /
``cuda`` / ``cuda:N`` / ``mps`` / ``mps:N``。显式指定不可用的加速器时立即退出，
而不是等到第一次分配显存才让下游抛出无信息量的 RuntimeError。
"""

import torch

AUTO = "auto"


def _cuda_ready() -> bool:
    return torch.cuda.is_available()


def _mps_ready() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


def _describe(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index or 0
        name = torch.cuda.get_device_name(index)
        return f"cuda:{index} ({name}, torch {torch.__version__})"
    if device.type == "mps":
        return f"mps ({device}, torch {torch.__version__})"
    return f"cpu (torch {torch.__version__})"


def resolve_device(spec: str) -> torch.device:
    """把 ``--device`` 字符串解析为一个经过可用性校验的 :class:`torch.device`。

    ``auto`` 依次尝试 CUDA、MPS，最后回落到 CPU。显式指定 ``cuda`` / ``mps``
    但当前不可用时抛 :class:`SystemExit`，错误信息给出可操作的修复方式。
    """
    if not isinstance(spec, str) or not spec.strip():
        raise SystemExit("错误：--device 不能为空（可选 auto/cpu/cuda/cuda:N/mps）")

    text = spec.strip()
    if text.lower() == AUTO:
        if _cuda_ready():
            return torch.device("cuda", 0)
        if _mps_ready():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        device = torch.device(text)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：无法解析 --device {spec!r}（可选 auto/cpu/cuda/cuda:N/mps）：{exc}")
    if device.type == "cuda" and not _cuda_ready():
        raise SystemExit(
            "错误：--device 指定的 CUDA 不可用。请确认已安装带 CUDA 的 torch"
            "（当前 torch {} 为 {} 版），或改用 --device cpu。".format(
                torch.__version__, "CPU" if not torch.cuda.is_available() else "CUDA"
            )
        )
    if device.type == "mps" and not _mps_ready():
        raise SystemExit("错误：--device 指定的 MPS 不可用（非 Apple 平台或版本不支持），请改用 --device cpu。")
    if device.type not in ("cpu", "cuda", "mps"):
        raise SystemExit(f"错误：不支持的设备类型 {device.type}，可选 auto/cpu/cuda/cuda:N/mps")
    return device


def enable_fast_math(device: torch.device) -> bool:
    """开启 TF32 matmul。返回是否实际生效。

    TF32 会降低 matmul 精度并改变数值，因此只在显式开启时调用，且仅对 CUDA
    生效——CPU 与 MPS 的精度设置不属于本管线的实验控制变量。
    """
    if device.type != "cuda":
        return False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        pass
    return True


def report_device(spec: str, tf32: bool = False) -> torch.device:
    """解析设备（仅此一次）、按需开启 TF32 并打印实际使用的设备。

    打印是为了避免“我以为在 GPU 上跑”的误判——设备探测失败会在这里直接退出，
    而不是在下游某个张量分配处抛出无信息量的 RuntimeError。返回解析好的设备，
    调用方（子命令入口）与训练函数共用同一解析结果即可，不要各自再 resolve。
    """
    device = resolve_device(spec)
    applied = enable_fast_math(device) if tf32 else False
    suffix = "  [tf32=on]" if applied else ""
    print(f"[device] {_describe(device)}{suffix}")
    return device
