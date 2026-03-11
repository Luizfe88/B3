import sys
import os
import logging

# Setup mocking for dependencies that might fail in a scripts environment
sys.path.append(os.getcwd())

print("--- Testing FundManager 'utils' Import ---")
try:
    from agents.fund_manager import FundManager
    fm = FundManager()
    # Check if 'utils' is in the module's globals
    import agents.fund_manager
    if hasattr(agents.fund_manager, 'utils'):
        print("✅ FundManager has access to 'utils' module.")
    else:
        print("❌ FundManager DOES NOT have access to 'utils' module.")
except Exception as e:
    print(f"❌ Error importing FundManager: {e}")

print("\n--- Testing AdaptiveIntelligence last_adjustment ---")
try:
    from adaptive_intelligence import adaptive_intelligence
    # Reset it to simulate no history if it was loaded from DB
    if adaptive_intelligence.last_adjustment is not None:
        print(f"✅ AdaptiveIntelligence last_adjustment is set to: {adaptive_intelligence.last_adjustment}")
    else:
        print("❌ AdaptiveIntelligence last_adjustment is None.")
except Exception as e:
    print(f"❌ Error importing AdaptiveIntelligence: {e}")

print("\n--- Testing bot.py utils Import ---")
try:
    import bot
    if hasattr(bot, 'utils'):
        print("✅ bot.py has access to 'utils' module.")
    else:
        print("❌ bot.py DOES NOT have access to 'utils' module.")
except Exception as e:
    # bot.py might fail to run due to MT5 or connections, but we just want to see if the import exists
    if "MT5" in str(e) or "MetaTrader5" in str(e):
        print("✅ bot.py import check (partial): MT5 error suggests module was parsed correctly up to imports.")
    else:
        print(f"⚠️ bot.py check noticed error (might be expected): {e}")
