import os
import sys

# Define os 3 ativos para teste
os.environ["XP3_TARGET_SYMBOLS"] = "PETR4,VALE3,ITUB4"

# Importa e executa a rotina semanal
from calibration_routine_weekly import run_weekly_calibration

if __name__ == "__main__":
    run_weekly_calibration()
