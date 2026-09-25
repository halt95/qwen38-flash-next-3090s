"""A fake ``pynvml`` for the topology selector's tests: enough of the nvidia-ml-py API for ``NvmlTopology``.

Put this directory first on PYTHONPATH and point ``FAKE_PYNVML_SPEC`` at a JSON file:

    {
      "init_error": false,                     # true: nvmlInit raises NVMLError
      "gpus": [{"uuid": "GPU-...", "bus_id": "00000000:01:00.0", "gen_max": 4, "gen_cur": 4,
                "width_max": 16, "width_cur": 16, "procs": 0, "mem_used_mib": 1}],
      "ancestor": {"0-1": 40},                 # NVML topology level per pair (indices into gpus); default 40 (NODE)
      "p2p": {"0-1": "raise"},                 # per pair: "raise" (query fails) or {"read": code, "write": code};
                                               # default 0 (OK) both ways
      "nvlink_links": 12,                      # links that answer a state query; the next one raises NotSupported
      "nvlink": {"0": {"0": "00000000:81:00.0"}}   # gpu -> link -> remote bus id of an ACTIVE link
    }

Reproduces the behaviours the selector has to cope with on real hosts:
* ``NVML_P2P_CAPS_INDEX_READ`` / ``_WRITE`` are 1-tuples, and ``nvmlDeviceGetP2PStatus`` raises
  ``ctypes.ArgumentError`` when handed a tuple (so the selector must unwrap them);
* ``nvmlDeviceGetNvLinkState`` returns 0 (inactive) on links of a card without NVLink rather than raising, and raises
  NotSupported past the last link;
* PCI bus ids come back as ``bytes``.
"""
from __future__ import annotations

import ctypes
import json
import os

NVML_P2P_CAPS_INDEX_READ = (0,)
NVML_P2P_CAPS_INDEX_WRITE = (1,)
NVML_P2P_STATUS_OK = 0
NVML_TOPOLOGY_INTERNAL = 0
NVML_TOPOLOGY_SINGLE = 10
NVML_TOPOLOGY_MULTIPLE = 20
NVML_TOPOLOGY_HOSTBRIDGE = 30
NVML_TOPOLOGY_NODE = 40
NVML_TOPOLOGY_SYSTEM = 50
NVML_FEATURE_ENABLED = 1
NVML_NVLINK_MAX_LINKS = 18
NVML_ERROR_INVALID_ARGUMENT = 2
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_UNKNOWN = 999


class NVMLError(Exception):
    def __init__(self, value: int) -> None:
        super().__init__(f"NVML error {value}")
        self.value = value


class NVMLError_NotSupported(NVMLError):
    def __init__(self) -> None:
        super().__init__(NVML_ERROR_NOT_SUPPORTED)


class _Obj:
    def __init__(self, **kw) -> None:
        self.__dict__.update(kw)


def _spec() -> dict:
    with open(os.environ["FAKE_PYNVML_SPEC"], encoding="utf-8") as f:
        return json.load(f)


_S: dict = {}


def _pair(a: int, b: int, table: str, default):
    t = _S.get(table) or {}
    return t.get(f"{a}-{b}", t.get(f"{b}-{a}", default))


def nvmlInit() -> None:
    _S.clear()
    _S.update(_spec())
    if _S.get("init_error"):
        raise NVMLError(NVML_ERROR_UNKNOWN)


def nvmlShutdown() -> None:
    pass


def nvmlDeviceGetCount() -> int:
    return len(_S["gpus"])


def nvmlDeviceGetHandleByIndex(i: int) -> int:
    return i


def nvmlDeviceGetUUID(h: int) -> str:
    return _S["gpus"][h]["uuid"]


def nvmlDeviceGetPciInfo(h: int):
    return _Obj(busId=_S["gpus"][h]["bus_id"].encode("ascii"))


def nvmlDeviceGetMaxPcieLinkGeneration(h: int) -> int:
    return _S["gpus"][h].get("gen_max", 4)


def nvmlDeviceGetCurrPcieLinkGeneration(h: int) -> int:
    return _S["gpus"][h].get("gen_cur", 4)


def nvmlDeviceGetMaxPcieLinkWidth(h: int) -> int:
    return _S["gpus"][h].get("width_max", 16)


def nvmlDeviceGetCurrPcieLinkWidth(h: int) -> int:
    return _S["gpus"][h].get("width_cur", 16)


def nvmlDeviceGetTopologyCommonAncestor(a: int, b: int) -> int:
    return _pair(a, b, "ancestor", NVML_TOPOLOGY_NODE)


def nvmlDeviceGetP2PStatus(a: int, b: int, index) -> int:
    if not isinstance(index, int):
        raise ctypes.ArgumentError("argument 3: <class 'TypeError'>: wrong type")
    p = _pair(a, b, "p2p", None)
    if p == "raise":
        raise NVMLError(NVML_ERROR_UNKNOWN)
    if p is None:
        return NVML_P2P_STATUS_OK
    return p["read"] if index == 0 else p["write"]


def nvmlDeviceGetNvLinkState(h: int, link: int) -> int:
    if link >= _S.get("nvlink_links", 12):
        raise NVMLError_NotSupported()
    return 1 if str(link) in (_S.get("nvlink") or {}).get(str(h), {}) else 0


def nvmlDeviceGetNvLinkRemotePciInfo_v2(h: int, link: int):
    bus = (_S.get("nvlink") or {}).get(str(h), {}).get(str(link))
    if bus is None:
        raise NVMLError(NVML_ERROR_INVALID_ARGUMENT)
    return _Obj(busId=bus.encode("ascii"))


def nvmlDeviceGetComputeRunningProcesses(h: int) -> list:
    return [object()] * _S["gpus"][h].get("procs", 0)


def nvmlDeviceGetMemoryInfo(h: int):
    return _Obj(used=_S["gpus"][h].get("mem_used_mib", 1) * 1024 * 1024)
