import time
from datetime import datetime
import traceback
import sys

# Importa o módulo principal que criamos
import agent_supervisor

def loop_agendador():
    # Horários que o usuário pediu para a IA analisar o bot
    horarios_alvo = ["11:00", "14:00", "17:00"]
    
    print("==================================================")
    print("AGENDADOR DO SUPERVISOR XP3 INICIADO")
    print(f"A Reuniao do Conselho com Gemini acontecera as: {', '.join(horarios_alvo)}")
    print("Mantenha esta janela aberta. Pressione Ctrl+C para sair.")
    print("==================================================")

    while True:
        try:
            agora = datetime.now()
            hora_atual_str = agora.strftime("%H:%M")
            dia_semana = agora.weekday() # 0 = Seg, 4 = Sex, 5 = Sab, 6 = Dom

            # Só opera de segunda a sexta (opcional, mas recomendado para mercado B3)
            if dia_semana >= 5:
                # Fim de semana, dorme por 1 hora
                time.sleep(3600)
                continue

            # Se for o horário exato e os segundos estiverem no comecinho do minuto
            if hora_atual_str in horarios_alvo and agora.second < 10:
                print(f"\n[{agora.strftime('%Y-%m-%d %H:%M:%S')}] ⏰ Horário atingido ({hora_atual_str})!")
                print("Invocando a Reunião do Conselho Multi-Agentes...")
                
                try:
                    # Chama a função principal do supervisor
                    agent_supervisor.main()
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Reunião concluída.")
                except Exception as e:
                    print(f"❌ Erro grave na execução do supervisor: {e}")
                    traceback.print_exc()

                # Dorme 61 segundos para garantir que não vai disparar duas vezes 
                # no mesmo minuto (ex: rodar as 11:00:05 e depois 11:00:15)
                time.sleep(61)
            else:
                # Dorme 5 segundos antes de checar de novo
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n⏹️ Agendador cancelado pelo usuário. Encerrando.")
            sys.exit(0)
        except Exception as err:
            print(f"Erro no loop principal do agendador: {err}")
            time.sleep(10) # Pausa de segurança

if __name__ == "__main__":
    loop_agendador()
