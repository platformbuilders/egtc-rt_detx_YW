# rtdetr_detector.py - Detector de pessoas usando RT-DETR-X
import os
import numpy as np
import cv2
import torch
from typing import List, Tuple, Optional

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

class RTDETRPerson:
    """
    Detector de pessoas usando RT-DETR-X.
    Tenta usar Ultralytics primeiro, depois PaddleDetection como fallback.
    """
    def __init__(self, weights: str = "rtdetr-x.pt", device: str = "cuda", 
                 imgsz: int = 640, conf: float = 0.25, iou: float = 0.45,
                 min_area: float = 0.0, max_area: float = 1.0,
                 min_aspect_ratio: float = 0.0, max_aspect_ratio: float = 10.0,
                 min_height_px: int = 0, min_width_px: int = 0,
                 disable_filters: bool = False, debug: bool = False):
        """
        Inicializa o detector RT-DETR-X.
        
        Args:
            weights: Caminho do modelo ou nome (rtdetr-x.pt, rtdetr-l.pt, etc)
            device: Dispositivo (cuda/cpu)
            imgsz: Tamanho da imagem para inferência
            conf: Threshold de confiança
            iou: Threshold de IoU para NMS
        """
        self.weights = weights
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_height_px = min_height_px
        self.min_width_px = min_width_px
        self.disable_filters = disable_filters
        self.debug = debug
        self.model = None
        self.use_ultralytics = False
        
        # Tenta carregar via Ultralytics (RT-DETR suportado desde v8.1.0)
        if ULTRALYTICS_AVAILABLE:
            try:
                self.model = YOLO(weights)
                self.model.to(device)
                self.use_ultralytics = True
                self.half_precision = (device == "cuda")
                self.model.overrides["conf"] = conf
                self.model.overrides["iou"] = iou
                self.model.overrides["agnostic_nms"] = False
                self.model.overrides["max_det"] = 300
                # Testa uma inferência simples para verificar se funciona
                import torch
                test_img = torch.zeros((3, 640, 640), dtype=torch.uint8).numpy().transpose(1, 2, 0)
                test_results = self.model.predict(test_img, verbose=False, classes=[0])
                print(f"[INFO] RT-DETR: Modelo carregado e testado com sucesso - {weights}")
                print(f"[INFO] RT-DETR: Dispositivo={device}, imgsz={imgsz}, conf={conf}, iou={iou}")
            except Exception as e:
                print(f"[WARN] RT-DETR: Falha ao carregar modelo '{weights}': {e}")
                print(f"[INFO] RT-DETR: RT-DETR pode não estar disponível nesta versão do Ultralytics")
                print(f"[INFO] RT-DETR: Tente usar YOLOv8 (yolov8m.pt, yolov8l.pt, yolov8x.pt) como alternativa")
                raise
        else:
            print("[WARN] Ultralytics não disponível. Tentando PaddleDetection...")
            self._load_paddledetection()
    
    def _load_paddledetection(self):
        """Fallback para PaddleDetection se Ultralytics não funcionar."""
        try:
            import paddle
            from paddle.inference import create_predictor, Config
            print("[INFO] RT-DETR: Usando PaddleDetection (requer instalação)")
            # Implementação PaddleDetection aqui se necessário
            raise NotImplementedError("PaddleDetection não implementado ainda. Use Ultralytics.")
        except ImportError:
            raise ImportError(
                "RT-DETR requer Ultralytics ou PaddleDetection. "
                "Instale: pip install ultralytics>=8.1.0"
            )
    
    def detect(self, frame_bgr: np.ndarray, debug: bool = False) -> List[Tuple[float, float, float, float, float]]:
        """
        Detecta pessoas no frame.
        
        Retorna:
            Lista de boxes: [(x1, y1, x2, y2, conf), ...]
        """
        if self.model is None:
            return []
        
        H, W = frame_bgr.shape[:2]
        frame_area = H * W
        
        if self.use_ultralytics:
            # RT-DETR via Ultralytics - usando a mesma abordagem simples que funciona no rt-test.py
            results = self.model.predict(
                source=frame_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                classes=[0],  # Apenas pessoas (classe 0 no COCO)
                verbose=False,
                device=self.device
            )
            
            boxes = []
            raw_count = 0
            filtered_by_size = 0
            filtered_by_area = 0
            filtered_by_aspect = 0
            
            if len(results) > 0 and results[0].boxes is not None:
                num_boxes = len(results[0].boxes)
                if debug and num_boxes > 0:
                    print(f"  [RT-DETR] Total de boxes retornados: {num_boxes}")
                
                for idx, b in enumerate(results[0].boxes):
                    raw_count += 1
                    try:
                        # Extrai coordenadas da mesma forma simples que funciona no rt-test.py
                        xyxy = b.xyxy[0].cpu().numpy()
                        conf = float(b.conf[0].item())
                        
                        x1, y1, x2, y2 = map(float, xyxy)
                        
                        # Valida dimensões (mesma validação do rt-test.py)
                        if x2 <= x1 or y2 <= y1:
                            if debug:
                                print(f"  [FILTRO] Box {idx} inválido: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}, conf={conf:.3f}")
                            filtered_by_size += 1
                            continue
                        
                        # Garante que está dentro do frame
                        x1 = max(0, min(x1, W-1))
                        y1 = max(0, min(y1, H-1))
                        x2 = max(x1+1, min(x2, W))
                        y2 = max(y1+1, min(y2, H))
                        
                        box_w = x2 - x1
                        box_h = y2 - y1
                        box_area = box_w * box_h
                        
                        # Aplica filtros se não estiverem desabilitados
                        if not self.disable_filters:
                            # Filtro 1: Tamanho mínimo
                            if box_h < self.min_height_px or box_w < self.min_width_px:
                                if debug:
                                    print(f"  [FILTRO] Box rejeitado por tamanho: h={box_h:.0f}px, w={box_w:.0f}px, conf={conf:.3f}")
                                filtered_by_size += 1
                                continue
                            
                            # Filtro 2: Área relativa
                            if self.min_area > 0 or self.max_area < 1.0:
                                relative_area = box_area / frame_area if frame_area > 0 else 0
                                if relative_area < self.min_area or relative_area > self.max_area:
                                    if debug:
                                        print(f"  [FILTRO] Box rejeitado por área: {relative_area:.4f}, conf={conf:.3f}")
                                    filtered_by_area += 1
                                    continue
                            
                            # Filtro 3: Aspect ratio
                            if box_w > 0:
                                aspect_ratio = box_h / box_w
                                if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                                    if debug:
                                        print(f"  [FILTRO] Box rejeitado por aspect ratio: {aspect_ratio:.2f}, conf={conf:.3f}")
                                    filtered_by_aspect += 1
                                    continue
                        
                        boxes.append((x1, y1, x2, y2, conf))
                    except Exception as e:
                        if debug:
                            print(f"  [ERRO] Falha ao processar box {idx}: {e}")
                        filtered_by_size += 1
                        continue
            
            if debug and raw_count > 0:
                print(f"  [RT-DETR DEBUG] Detecções brutas: {raw_count}, Aceitas: {len(boxes)}, "
                      f"Filtradas (tamanho: {filtered_by_size}, área: {filtered_by_area}, aspect: {filtered_by_aspect})")
            
            return boxes
        
        return []

