"""Cache-free entry point, including when Python is invoked without -B."""
import sys
sys.dont_write_bytecode = True
from point_select.cli import main

if __name__ == "__main__":
    main()
