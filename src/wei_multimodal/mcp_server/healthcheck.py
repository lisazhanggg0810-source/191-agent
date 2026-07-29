"""Container readiness probe without credentials or response-body logging."""

from __future__ import annotations

import sys
from urllib.error import URLError
from urllib.request import urlopen


def main() -> int:
    """Return zero only when the local readiness endpoint responds with HTTP 200."""

    try:
        with urlopen("http://127.0.0.1:8000/health/ready", timeout=3) as response:
            return 0 if response.status == 200 else 1
    except (OSError, URLError):
        return 1


if __name__ == "__main__":
    sys.exit(main())
