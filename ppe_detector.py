# ppe_detector.py - Wrapper unificado para YOLO-World e OWL-V2
import os
import sys
from typing import Dict, List, Tuple, Optional
import numpy as np

# Importa YOLO-World (local)
from yolo_world_ppe import YOLOWorldPPE

# Tenta importar OWL-V2 (do mesmo diretório do executável)
# A importação será feita dinamicamente quando necessário
_OWL_V2_ERROR_MSG = None

def _check_owl_v2_available():
    """Verifica se OWL-V2 está disponível e retorna o módulo se estiver."""
    global _OWL_V2_ERROR_MSG
    try:
        # Primeiro verifica se transformers está disponível
        try:
            import transformers
        except ImportError:
            _OWL_V2_ERROR_MSG = "Módulo 'transformers' não encontrado. Instale com: pip install transformers"
            print(f"[WARN] OWL-V2 não disponível: {_OWL_V2_ERROR_MSG}")
            return None
        
        # Tenta importar diretamente (Python encontrará se estiver no mesmo diretório ou PYTHONPATH)
        from ovd import OpenVocabPPE
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"[INFO] OWL-V2 disponível (carregado de {script_dir})")
        _OWL_V2_ERROR_MSG = None
        return OpenVocabPPE
    except ImportError as e:
        _OWL_V2_ERROR_MSG = f"Erro ao importar OWL-V2: {e}. Verifique se ovd.py existe no mesmo diretório."
        print(f"[WARN] OWL-V2 não disponível: {_OWL_V2_ERROR_MSG}")
        return None
    except Exception as e:
        _OWL_V2_ERROR_MSG = f"Erro ao carregar OWL-V2: {e}"
        print(f"[WARN] OWL-V2 não disponível: {_OWL_V2_ERROR_MSG}")
        return None

# Tenta carregar na inicialização do módulo
OpenVocabPPE = _check_owl_v2_available()
OWL_V2_AVAILABLE = OpenVocabPPE is not None

class UnifiedPPEDetector:
    """
    Wrapper unificado para YOLO-World e OWL-V2.
    Permite alternar entre os dois modelos via flag.
    """
    
    def __init__(self, detector_type: str = "yolo-world", **kwargs):
        """
        Inicializa o detector unificado.
        
        Args:
            detector_type: "yolo-world" ou "owl-v2"
            **kwargs: Parâmetros específicos de cada detector
        """
        self.detector_type = detector_type.lower()
        self.detector = None
        self.use_crop = False  # Para compatibilidade
        
        if self.detector_type == "yolo-world":
            # Parâmetros YOLO-World
            model_name = kwargs.get("model_name", kwargs.get("yw_model", "yolov8m-world.pt"))
            device = kwargs.get("device", "cuda")
            fp16 = kwargs.get("fp16", kwargs.get("yw_fp16", True))
            use_crop = kwargs.get("use_crop", kwargs.get("yw_use_crop", False))
            crop_padding = kwargs.get("crop_padding", kwargs.get("yw_crop_padding", 0.15))
            min_crop_size = kwargs.get("min_crop_size", kwargs.get("yw_min_crop_size", 32))
            imgsz = kwargs.get("imgsz", kwargs.get("yw_imgsz", 640))
            
            self.detector = YOLOWorldPPE(
                model_name=model_name,
                device=device,
                fp16=fp16,
                use_crop=use_crop,
                crop_padding=crop_padding,
                min_crop_size=min_crop_size,
                imgsz=imgsz
            )
            self.use_crop = use_crop
            
        elif self.detector_type == "owl-v2":
            # Tenta carregar OWL-V2 se ainda não foi carregado
            global OpenVocabPPE, OWL_V2_AVAILABLE, _OWL_V2_ERROR_MSG
            if not OWL_V2_AVAILABLE:
                OpenVocabPPE_module = _check_owl_v2_available()
                if OpenVocabPPE_module is None:
                    error_msg = _OWL_V2_ERROR_MSG or "OWL-V2 não está disponível. Verifique se ovd.py existe no mesmo diretório do pipeline."
                    raise ImportError(
                        f"{error_msg}\n"
                        f"Solução: Instale as dependências com 'pip install transformers torch pillow' ou use 'ppe_detector: yolo-world' no YAML."
                    )
                # Atualiza a referência global
                OpenVocabPPE = OpenVocabPPE_module
                OWL_V2_AVAILABLE = True
            
            # Parâmetros OWL-V2
            model_name = kwargs.get("model_name", kwargs.get("ovd_model", "google/owlv2-base-patch16"))
            device = kwargs.get("device", "cuda")
            fp16 = kwargs.get("fp16", kwargs.get("ovd_fp16", True))
            cache_dir = kwargs.get("cache_dir", kwargs.get("ovd_cache_dir", "./.hf"))
            use_fast = kwargs.get("use_fast", kwargs.get("ovd_use_fast", True))
            quantization_mode = kwargs.get("quantization_mode", kwargs.get("ovd_quantization_mode", "none"))
            
            self.detector = OpenVocabPPE(
                model_name=model_name,
                device=device,
                fp16=fp16,
                cache_dir=cache_dir,
                use_fast=use_fast,
                quantization_mode=quantization_mode
            )
            # OWL-V2 não suporta modo crop (sempre processa frame completo)
            self.use_crop = False
            
        else:
            raise ValueError(f"Tipo de detector não suportado: {detector_type}. Use 'yolo-world' ou 'owl-v2'")
        
        print(f"[INFO] Detector PPE: {self.detector_type.upper()} inicializado")
    
    def infer(self, image_bgr: np.ndarray, prompts: Dict[str, List[str]], 
              score_thr: float = 0.25, person_boxes: Optional[List[List[int]]] = None,
              negative: Optional[Dict[str, List[str]]] = None) -> Tuple[Optional[Dict], Dict[str, List[Tuple[float, List[float]]]]]:
        """
        Inferência unificada.
        
        Args:
            image_bgr: Imagem BGR (numpy array)
            prompts: Dicionário com prompts positivos (ex: {"helmet": ["helmet", "hard hat"], ...})
            score_thr: Threshold de confiança
            person_boxes: Lista de boxes de pessoas para modo crop (apenas YOLO-World)
            negative: Prompts negativos (apenas OWL-V2)
        
        Returns:
            (flags_or_classes, raw_detections) onde:
            - flags_or_classes: Dict para OWL-V2, None para YOLO-World (não usado)
            - raw_detections: {label: [(score, [x1,y1,x2,y2]), ...]}
        """
        if self.detector_type == "yolo-world":
            # YOLO-World: retorna (classes, raw)
            classes, raw = self.detector.infer(
                image_bgr, 
                prompts, 
                score_thr=score_thr,
                person_boxes=person_boxes if self.use_crop else None
            )
            # Normaliza retorno: retorna None no lugar de classes (não usado no pipeline)
            return None, raw
            
        elif self.detector_type == "owl-v2":
            # OWL-V2: precisa de positive e negative separados
            positive = prompts
            negative = negative or {}
            
            # OWL-V2 não suporta person_boxes (sempre processa frame completo)
            flags, raw = self.detector.infer(
                image_bgr,
                positive,
                negative,
                score_thr=score_thr
            )
            # Retorna flags (não usado no pipeline, mas mantido para compatibilidade)
            return flags, raw
        
        else:
            raise ValueError(f"Tipo de detector inválido: {self.detector_type}")

