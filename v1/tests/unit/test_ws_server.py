from __future__ import annotations

import sys
import types
from unittest.mock import patch

fake_numpy = types.ModuleType("numpy")
setattr(fake_numpy, "array", lambda *args, **kwargs: [])
setattr(fake_numpy, "zeros", lambda *args, **kwargs: [])
setattr(fake_numpy, "linspace", lambda *args, **kwargs: [])
setattr(fake_numpy, "meshgrid", lambda *args, **kwargs: ([], []))
setattr(fake_numpy, "exp", lambda *args, **kwargs: 0)
setattr(fake_numpy, "sqrt", lambda *args, **kwargs: 0)
setattr(fake_numpy, "clip", lambda value, *_args, **_kwargs: value)
setattr(fake_numpy, "float64", float)
setattr(fake_numpy, "ndarray", list)
sys.modules.setdefault("numpy", fake_numpy)

fake_numpy_typing = types.ModuleType("numpy.typing")
setattr(fake_numpy_typing, "NDArray", list)
sys.modules.setdefault("numpy.typing", fake_numpy_typing)

fake_scipy = types.ModuleType("scipy")
setattr(fake_scipy, "fft", types.SimpleNamespace())
setattr(fake_scipy, "stats", types.SimpleNamespace())
sys.modules.setdefault("scipy", fake_scipy)

from v1.src.sensing.ws_server import SensingWebSocketServer


def test_create_collector_falls_back_when_linux_wifi_probe_fails() -> None:
    server = SensingWebSocketServer()
    simulated_collector = object()

    with (
        patch("v1.src.sensing.ws_server.probe_esp32_udp", return_value=False),
        patch("v1.src.sensing.ws_server.platform.system", return_value="Linux"),
        patch("os.path.exists", return_value=True),
        patch(
            "v1.src.sensing.ws_server.LinuxWifiCollector.collect_once",
            side_effect=RuntimeError("wifi unavailable"),
        ),
        patch(
            "v1.src.sensing.ws_server.SimulatedCollector",
            return_value=simulated_collector,
        ),
    ):
        collector = server._create_collector()

    assert collector is simulated_collector
    assert server.source == "simulated"
