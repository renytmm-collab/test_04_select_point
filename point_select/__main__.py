import sys

# Every command, including read-only commands, forbids bytecode side effects.
sys.dont_write_bytecode = True
from .cli import main

main()
