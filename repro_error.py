import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

try:
    from core.execution import ExecutionEngine
    e = ExecutionEngine()
    print(f"ExecutionEngine has get_account_info: {hasattr(e, 'get_account_info')}")
    if hasattr(e, 'get_account_info'):
        print("Success! get_account_info exists.")
    else:
        print("Failure! get_account_info DOES NOT exist.")
        print(f"Attributes: {dir(e)}")
except Exception as err:
    print(f"Error: {err}")
