
import sys
import unittest
from datetime import datetime

# Simulação de como o FundManager faz o check
def simulated_check(current_datetime, config_friday_no_entry):
    now = current_datetime
    current_time = now.time()
    
    if now.weekday() == 4: # Sexta-feira
         no_entry_str = config_friday_no_entry
    else:
         no_entry_str = "16:15" # Padrão
         
    no_entry_time = datetime.strptime(no_entry_str, "%H:%M").time()
    
    if current_time >= no_entry_time:
         return False, f"Entry Cutoff Time Reached ({no_entry_str})"
    return True, "Entry Allowed"

class TestFridayRestrictionsSimplified(unittest.TestCase):
    def test_friday_after_3pm(self):
        # 6 de Março de 2026 é uma Sexta-feira (weekday == 4)
        friday_after_3pm = datetime(2026, 3, 6, 15, 10, 0)
        config_val = "15:00"
        
        allowed, reason = simulated_check(friday_after_3pm, config_val)
        print(f"At {friday_after_3pm}, config={config_val}: Allowed={allowed}, Reason={reason}")
        self.assertFalse(allowed)
        self.assertIn("15:00", reason)

    def test_friday_before_3pm(self):
        friday_before_3pm = datetime(2026, 3, 6, 14, 50, 0)
        config_val = "15:00"
        
        allowed, reason = simulated_check(friday_before_3pm, config_val)
        print(f"At {friday_before_3pm}, config={config_val}: Allowed={allowed}, Reason={reason}")
        self.assertTrue(allowed)

    def test_thursday_after_3pm(self):
        # 5 de Março de 2026 é uma Quinta-feira (weekday == 3)
        thursday_after_3pm = datetime(2026, 3, 5, 15, 10, 0)
        config_val = "15:00" # Friday specific, Thursday should use standard 16:15
        
        allowed, reason = simulated_check(thursday_after_3pm, config_val)
        print(f"At {thursday_after_3pm}, config={config_val}: Allowed={allowed}, Reason={reason}")
        self.assertTrue(allowed)

if __name__ == "__main__":
    unittest.main()
