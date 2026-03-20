import sys
import os

print(f"Current Directory: {os.getcwd()}")
print(f"sys.path: {sys.path}")

try:
    import core.execution
    print(f"core.execution file: {core.execution.__file__}")
    from core.execution import ExecutionEngine
    e = ExecutionEngine()
    print(f"ExecutionEngine attributes: {[attr for attr in dir(e) if not attr.startswith('__')]}")
    print(f"Has get_account_info: {hasattr(e, 'get_account_info')}")
except Exception as err:
    print(f"Error: {err}")
