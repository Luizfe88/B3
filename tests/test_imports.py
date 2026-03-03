
import os
import sys
import logging

# Setup paths
root_path = os.path.abspath(os.getcwd())
if root_path not in sys.path:
    sys.path.append(root_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestImports")

def test_analyst_imports():
    try:
        from agents.analyst_team import AnalystTeam
        team = AnalystTeam()
        logger.info("✅ AnalystTeam instanciado com sucesso.")
        
        from fundamentals import fundamental_fetcher
        logger.info("✅ FundamentalFetcher (global) importado com sucesso.")
        
        # Simula uma chamada rápida para ver se 'utils' é reconhecido
        # Não chamamos mt5.initialize aqui para evitar dependência de terminal aberto
        # mas verificamos se o nome 'utils' existe no escopo do módulo
        import agents.analyst_team as at
        if hasattr(at, 'utils'):
            logger.info("✅ 'utils' encontrado no escopo de analyst_team.py")
        else:
            raise NameError("'utils' não encontrado no escopo global de analyst_team.py")
            
        import fundamentals as fund
        if hasattr(fund, 'utils'):
            logger.info("✅ 'utils' encontrado no escopo de fundamentals.py")
        else:
            raise NameError("'utils' não encontrado no escopo global de fundamentals.py")

        logger.info("🚀 Todos os testes de importação passaram!")
        
    except Exception as e:
        logger.error(f"❌ Falha no teste de importação: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_analyst_imports()
