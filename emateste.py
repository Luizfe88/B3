import json
import os
import glob


def consolidar_debug():
    # 1. Caminhos - Verifique se estes caminhos existem no seu Windows
    pasta_origem = r"C:\Users\luizf\Documents\xp3v5\optimizer_output\novoema"
    arquivo_final = os.path.join(pasta_origem, "resultado_consolidado.json")

    print(f"--- Iniciando Consolidação ---")
    print(f"Procurando ficheiros em: {pasta_origem}")

    # 2. Verifica se a pasta existe
    if not os.path.exists(pasta_origem):
        print(f"ERRO: A pasta {pasta_origem} não existe!")
        return

    # 3. Procura ficheiros _history.json
    padrao = os.path.join(pasta_origem, "*_history.json")
    arquivos = glob.glob(padrao)

    print(f"Ficheiros encontrados: {len(arquivos)}")

    if len(arquivos) == 0:
        print(
            "AVISO: Nenhum ficheiro '_history.json' foi encontrado. O JSON virá vazio."
        )
        return

    consolidado = {}

    # 4. Processa cada ficheiro
    for caminho in arquivos:
        nome_base = os.path.basename(caminho)
        simbolo = nome_base.replace("_history.json", "")

        try:
            with open(caminho, "r", encoding="utf-8") as f:
                dados = json.load(f)

                # Extrai as métricas (como as 43 trades de ITUB4)
                consolidado[simbolo] = {
                    "metricas": dados.get("metrics", {}),
                    "parametros": dados.get("best", {}),
                    "data": dados.get("generated_at"),
                }
                print(f"✅ {simbolo} adicionado.")
        except Exception as e:
            print(f"❌ Erro ao ler {nome_base}: {e}")

    # 5. Grava o resultado final
    try:
        with open(arquivo_final, "w", encoding="utf-8") as f:
            json.dump(consolidado, f, indent=4, ensure_ascii=False)
        print(f"---")
        print(f"SUCESSO: Ficheiro gerado em {arquivo_final}")
    except Exception as e:
        print(f"ERRO ao gravar ficheiro final: {e}")


if __name__ == "__main__":
    consolidar_debug()
