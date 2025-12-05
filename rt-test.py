#!/usr/bin/env python3
"""
Script simples para testar RT-DETR-X na detecção de pessoas.
Apenas detecta pessoas e mostra o vídeo na tela.
"""
import cv2
import yaml
from ultralytics import YOLO
import numpy as np

def load_config(config_path):
    """Carrega configuração do YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Carrega configuração
    config = load_config('rt-test.yaml')
    camera_url = config['camera_url']
    model_name = config.get('model', 'rtdetr-x.pt')
    conf_threshold = config.get('conf', 0.25)
    imgsz = config.get('imgsz', 640)
    device = config.get('device', 'cuda')
    
    print(f"[INFO] Carregando modelo: {model_name}")
    print(f"[INFO] URL da câmera: {camera_url}")
    print(f"[INFO] Confiança: {conf_threshold}, imgsz: {imgsz}, device: {device}")
    
    # Carrega modelo RT-DETR
    model = YOLO(model_name)
    model.to(device)
    
    # Abre câmera
    print(f"[INFO] Conectando à câmera...")
    cap = cv2.VideoCapture(camera_url)
    
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a câmera: {camera_url}")
        return
    
    print(f"[INFO] Câmera conectada. Pressione 'q' para sair.")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] Falha ao ler frame")
                continue
            
            frame_count += 1
            H, W = frame.shape[:2]
            
            # Detecta pessoas
            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf_threshold,
                classes=[0],  # Apenas pessoas (classe 0)
                verbose=False,
                device=device
            )
            
            # Processa resultados
            num_detections = 0
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                num_detections = len(boxes)
                
                # Desenha boxes
                for idx, box in enumerate(boxes):
                    try:
                        # Extrai coordenadas
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].item())
                        
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        # Valida dimensões
                        if x2 > x1 and y2 > y1:
                            # Desenha box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Desenha label
                            label = f"Person {conf:.2f}"
                            cv2.putText(frame, label, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            num_detections += 1
                        else:
                            print(f"  [WARN] Box {idx} inválido: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    except Exception as e:
                        print(f"  [ERRO] Falha ao processar box {idx}: {e}")
            
            # Mostra estatísticas
            info_text = f"Frame: {frame_count} | Detections: {num_detections} | Size: {W}x{H}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostra frame
            cv2.imshow('RT-DETR Test', frame)
            
            # Debug a cada 30 frames
            if frame_count % 30 == 0:
                print(f"[INFO] Frame {frame_count}: {num_detections} pessoa(s) detectada(s)")
            
            # Verifica tecla 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuário")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Finalizado. Total de frames processados: {frame_count}")

if __name__ == "__main__":
    main()

