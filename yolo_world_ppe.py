# yolo_world_ppe.py - Detector de EPIs usando YOLO-World
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from ultralytics import YOLOWorld
    YOLOWORLD_AVAILABLE = True
except ImportError:
    YOLOWORLD_AVAILABLE = False
    YOLOWorld = None

class YOLOWorldPPE:
    """
    Detector de EPIs usando YOLO-World (open-vocabulary).
    Detecta capacete, luvas, colete refletivo e protetores auriculares.
    """
    def __init__(self, model_name: str = "yolov8s-world.pt", device: str = "cuda",
                 fp16: bool = True, use_crop: bool = True, crop_padding: float = 0.15,
                 min_crop_size: int = 32, imgsz: int = 640):
        """
        Inicializa o detector YOLO-World para EPIs.
        
        Args:
            model_name: Nome do modelo YOLO-World (yolov8s-world.pt, yolov8m-world.pt, etc)
            device: Dispositivo (cuda/cpu)
            fp16: Usar precisão half (FP16)
            use_crop: Processar crops individuais de pessoas (melhor precisão)
            crop_padding: Padding relativo ao redor do crop
            min_crop_size: Tamanho mínimo do crop em pixels
        """
        if not YOLOWORLD_AVAILABLE:
            raise ImportError(
                "YOLO-World requer Ultralytics. Instale: pip install ultralytics>=8.2.0"
            )
        
        self.model = YOLOWorld(model_name)
        self.model.to(device)
        self.device = device
        self.fp16 = fp16
        self.use_crop = use_crop
        self.crop_padding = crop_padding
        self.min_crop_size = min_crop_size
        self.imgsz = imgsz
        self._last_classes = None
        
        print(f"[INFO] YOLO-World: Modelo carregado - {model_name}")
        print(f"[INFO] YOLO-World: Dispositivo={device}, fp16={fp16}, use_crop={use_crop}, imgsz={imgsz}, min_crop_size={min_crop_size}")
    
    @staticmethod
    def _flatten_prompts(prompts: dict) -> List[str]:
        """Achata os prompts em uma lista única de classes."""
        classes = []
        for _, synonyms in (prompts or {}).items():
            for s in (synonyms or []):
                if s and s not in classes:
                    classes.append(s)
        return classes
    
    def infer(self, image_bgr: np.ndarray, prompts: dict, score_thr: float = 0.25,
              person_boxes: Optional[List[List[int]]] = None) -> Tuple[List[str], Dict[str, List[Tuple[float, List[float]]]]]:
        """
        Infere EPIs na imagem.
        
        Args:
            image_bgr: Imagem BGR
            prompts: Dicionário com prompts positivos (ex: {"helmet": ["helmet", "hard hat"], ...})
            score_thr: Threshold de confiança
            person_boxes: Lista de boxes de pessoas [(x1, y1, x2, y2), ...] para modo crop
        
        Retorna:
            (classes, raw_detections) onde raw_detections é {label: [(score, [x1,y1,x2,y2]), ...]}
        """
        classes = self._flatten_prompts(prompts)
        if not classes:
            return [], {}
        
        # Atualiza classes se mudaram
        if classes != self._last_classes:
            self.model.set_classes(classes)
            self._last_classes = classes
        
        raw = {}
        H, W = image_bgr.shape[:2]
        
        # Modo crop: processa cada pessoa individualmente (melhor precisão)
        if self.use_crop and person_boxes and len(person_boxes) > 0:
            for pbox in person_boxes:
                x1, y1, x2, y2 = map(int, pbox)
                # Adiciona padding
                pad_w = int((x2 - x1) * self.crop_padding)
                pad_h = int((y2 - y1) * self.crop_padding)
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(W, x2 + pad_w)
                y2 = min(H, y2 + pad_h)
                
                # Valida tamanho mínimo
                if (x2 - x1) < self.min_crop_size or (y2 - y1) < self.min_crop_size:
                    continue
                
                crop = image_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # Converte BGR para RGB
                img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
                # Inferência com imgsz para melhor detecção de objetos pequenos
                # Usa imgsz maior que o crop para melhorar a detecção
                crop_h, crop_w = crop.shape[:2]
                # Calcula imgsz dinâmico baseado no tamanho do crop, mas com mínimo
                dynamic_imgsz = max(self.imgsz, min(crop_w, crop_h) * 2)
                # Limita a um máximo razoável para não sobrecarregar
                dynamic_imgsz = min(dynamic_imgsz, 1280)
                
                results = self.model.predict(
                    img_rgb,
                    imgsz=dynamic_imgsz,
                    conf=score_thr,
                    verbose=False,
                    stream=False,
                    half=self.fp16 if self.device == "cuda" else False
                )
                
                if results and len(results) > 0:
                    r0 = results[0]
                    if getattr(r0, "boxes", None) is not None:
                        boxes = r0.boxes
                        xyxy = boxes.xyxy.cpu().numpy()
                        scores = boxes.conf.cpu().numpy()
                        cls_idx = boxes.cls.cpu().numpy().astype(int)
                        
                        for (cx1, cy1, cx2, cy2), sc, ci in zip(xyxy, scores, cls_idx):
                            # Converte coordenadas do crop para coordenadas do frame original
                            orig_x1 = float(cx1 + x1)
                            orig_y1 = float(cy1 + y1)
                            orig_x2 = float(cx2 + x1)
                            orig_y2 = float(cy2 + y1)
                            
                            label = classes[ci] if 0 <= ci < len(classes) else "unknown"
                            raw.setdefault(label, []).append((float(sc), [orig_x1, orig_y1, orig_x2, orig_y2]))
        else:
            # Modo frame completo
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self.model.predict(
                img_rgb,
                imgsz=self.imgsz,
                conf=score_thr,
                verbose=False,
                stream=False,
                half=self.fp16 if self.device == "cuda" else False
            )
            
            if results and len(results) > 0:
                r0 = results[0]
                if getattr(r0, "boxes", None) is not None:
                    boxes = r0.boxes
                    xyxy = boxes.xyxy.cpu().numpy()
                    scores = boxes.conf.cpu().numpy()
                    cls_idx = boxes.cls.cpu().numpy().astype(int)
                    
                    for (x1, y1, x2, y2), sc, ci in zip(xyxy, scores, cls_idx):
                        label = classes[ci] if 0 <= ci < len(classes) else "unknown"
                        raw.setdefault(label, []).append((float(sc), [float(x1), float(y1), float(x2), float(y2)]))
        
        return classes, raw

