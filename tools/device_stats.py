"""pytest plugin (``-p tools.device_stats``): record how many loops each test
process really ran on a GPU, appended to ``$PARAKEET_DEVICE_STATS``."""

import os


def pytest_sessionfinish(session):
    path = os.environ.get("PARAKEET_DEVICE_STATS")
    if not path:
        return
    from parakeet.backends import cuda, metal

    for cls in (cuda.CudaDevice, metal.MetalDevice):
        dev = cls._instance
        if dev is not None:
            with open(path, "a") as f:
                f.write(f"{cls.name} {dev.device_name!r} launches={dev.launches}\n")
