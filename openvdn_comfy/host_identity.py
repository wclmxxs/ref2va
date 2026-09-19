"""Boot-bound process ownership; model and compiler caches remain portable."""
from functools import lru_cache
from pathlib import Path
import platform


@lru_cache(maxsize=1)
def host_identity():
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None
    boot = read('/proc/sys/kernel/random/boot_id')
    if not boot:
        import psutil
        boot = str(psutil.boot_time())
    return {'boot_id': boot,
            'machine_id': read('/sys/class/dmi/id/product_uuid') or read('/etc/machine-id') or platform.node()}


def belongs_to_host(record):
    # Legacy records also undergo PID birth-time/command checks at their callers.
    return 'host' not in record or record['host'] == host_identity()
