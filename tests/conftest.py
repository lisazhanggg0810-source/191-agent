"""Keep test imports from mutating the release source tree."""

import sys

sys.dont_write_bytecode = True
