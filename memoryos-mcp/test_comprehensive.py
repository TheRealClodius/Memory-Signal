#!/usr/bin/env python3
"""
Comprehensive test wrapper for MemoryOS MCP server.
Currently reuses SimpleMemoryOSTest to validate insertion and retrieval.

Run: python test_comprehensive.py
"""

import sys
import asyncio
from test_simple import SimpleMemoryOSTest


def main():
    tester = SimpleMemoryOSTest(server_script="server_new.py", config_file="config.json")
    ok = asyncio.run(tester.run_test())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


