# rtsp_test.py  ← VERSÃO CORRIGIDA E MELHORADA (2025)
import cv2
import os
import sys
import time
import argparse

parser = argparse.ArgumentParser(description="Teste RTSP forçando TCP ou UDP")
parser.add_argument("-p", "--protocol", choices=["udp", "tcp", "auto"], default="udp",
                    help="udp = força UDP | tcp = força TCP | auto = tenta UDP → TCP")
parser.add_argument("-u", "--url", type=str,
                    default="rtsp://admin:nsa_sintese2025@10.163.8.63:554/Streaming/Channels/101/",
                    help="URL RTSP")
args = parser.parse_args()

# ==================== FORÇA O PROTOCOLO ====================
if args.protocol == "udp":
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    modo = "UDP FORÇADO"
elif args.protocol == "tcp":
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    modo = "TCP FORÇADO"
else:  # auto
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    modo = "AUTO (UDP → TCP se falhar)"

print(f"\n{modo}")
print(f"URL: {args.url}\n")
print("-" * 70)

cv2.namedWindow(f"RTSP - {args.protocol.upper()} | q = sair", cv2.WINDOW_NORMAL)

tentativa = 1
while True:
    print(f"Tentativa {tentativa:4d} | {time.strftime('%H:%M:%S')} | Conectando...", end=" ")

    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("FALHOU")
        if args.protocol == "udp":
            print("    → UDP está bloqueado ou com perda de pacotes")
        time.sleep(2)
        tentativa += 1
        continue

    # Configurações anti-latência
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("CONECTADO! Exibindo stream...")
    conectado = True

    while conectado:
        ret, frame = cap.read()
        if not ret:
            print("Perda de frame/conexão → reconectando...")
            cap.release()
            time.sleep(1)
            conectado = False
            break

        # Overlay informativo
        cv2.putText(frame, f"MODO: {modo}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if args.protocol == "udp":
            cv2.putText(frame, "UDP FUNCIONANDO!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        elif args.protocol == "tcp":
            cv2.putText(frame, "TCP (fallback)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        cv2.putText(frame, f"Tentativa {tentativa} | {time.strftime('%H:%M:%S')}", (10, frame.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(f"RTSP - {args.protocol.upper()} | q = sair", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("\nScript encerrado pelo usuário.")
            sys.exit(0)

    cap.release()
    tentativa += 1
