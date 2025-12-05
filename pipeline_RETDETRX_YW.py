# pipeline_RETDETRX_YW.py — EGTC: RT-DETR-X (pessoas) + YOLO-World (EPIs), ROI, violação destacada, métricas
import os, time, yaml, cv2, csv, json, argparse, traceback
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
from collections import Counter

# Adiciona diretórios ao path para importar módulos
script_dir = os.path.dirname(os.path.abspath(__file__))
# IMPORTANTE: script_dir primeiro para garantir que utils.py local seja usado
sys.path.insert(0, script_dir)  # Para rtdetr_detector, yolo_world_ppe e utils local
# Adiciona diretório do projeto original para tracker (mas não para utils)
egtc_olm_dir = os.path.join(os.path.dirname(script_dir), 'egtc_olm')
if os.path.exists(egtc_olm_dir):
    # Adiciona ao final para que utils local tenha prioridade
    sys.path.append(egtc_olm_dir)

from rtdetr_detector import RTDETRPerson
from ppe_detector import UnifiedPPEDetector
# Importa utils do diretório local (egtc_detr) que tem as funções atualizadas
from utils import draw_person_box, draw_ppe_panel, put_banner, clamp_box, draw_rois, draw_alert_grid, GREEN, RED, YELLOW, ORANGE
from tracker import PPETracker
from alerts import AlertManager, AlertConfig

# --------------------- util de config ---------------------
def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_prompts(path: str):
    data = load_yaml(path)
    return data.get("positive", {}), data.get("negative", {})

# ----------------- helpers p/ atribuição YOLO-World ----------------
def _center(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def _inside(px, py, box):
    x1, y1, x2, y2 = box
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)

def _in_vertical_zone(py, person_box, zmin, zmax):
    x1, y1, x2, y2 = person_box
    h = (y2 - y1)
    return (py >= y1 + zmin * h) and (py <= y1 + zmax * h)

def _best_hit_in_group(raw_frame: Dict[str, List[Tuple[float, List[float]]]],
                       synonyms: List[str],
                       person_box: List[int],
                       zone: Optional[Tuple[float, float]]):
    best_score = -1.0
    best_box = None
    present = False
    for s in synonyms:
        for score, b in raw_frame.get(s, []):
            cx, cy = _center(b)
            if not _inside(cx, cy, person_box):
                continue
            if zone is not None and not _in_vertical_zone(cy, person_box, zone[0], zone[1]):
                continue
            if score > best_score:
                best_score = score
                best_box = b
                present = True
    return present, best_score, best_box

def _detect_helmet_color_from_pixels(frame_bgr: np.ndarray, helmet_box: List[float]) -> str:
    """
    Detecta a cor do capacete analisando os pixels do box detectado.
    Usa análise HSV para ser mais robusto a variações de iluminação.
    
    Args:
        frame_bgr: Frame BGR
        helmet_box: [x1, y1, x2, y2] do capacete detectado
    
    Returns:
        Cor detectada: "red", "blue", "yellow", "white", "brown", ou "-" se não conseguir determinar
    """
    x1, y1, x2, y2 = map(int, helmet_box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_bgr.shape[1], x2)
    y2 = min(frame_bgr.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return "-"
    
    # Extrai a região do capacete (foca no topo 60% para evitar fundo)
    box_h = y2 - y1
    crop_y1 = y1
    crop_y2 = y1 + int(box_h * 0.6)  # Topo 60% do box
    crop_y2 = min(crop_y2, y2)
    
    # Adiciona padding interno para evitar bordas
    pad = max(2, int(min(x2-x1, crop_y2-crop_y1) * 0.1))
    crop_x1 = x1 + pad
    crop_x2 = x2 - pad
    crop_y1 = crop_y1 + pad
    crop_y2 = crop_y2 - pad
    
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return "-"
    
    helmet_roi = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    if helmet_roi.size == 0:
        return "-"
    
    # Converte para HSV (mais robusto para análise de cor)
    hsv = cv2.cvtColor(helmet_roi, cv2.COLOR_BGR2HSV)
    
    # Remove pixels muito escuros ou muito claros (sombra/reflexo)
    # mask é 2D (height, width)
    mask = (hsv[:,:,2] > 30) & (hsv[:,:,2] < 240)  # V (brightness) entre 30-240
    
    if mask.sum() < 10:  # Muito poucos pixels válidos
        return "-"
    
    # Extrai arrays 2D completos para análise
    hues_2d = hsv[:,:,0]  # 2D
    saturations_2d = hsv[:,:,1]  # 2D
    values_2d = hsv[:,:,2]  # 2D
    
    # Calcula cor dominante baseado em ranges HSV
    # Ranges HSV (OpenCV usa H: 0-179, S: 0-255, V: 0-255)
    # IMPORTANTE: Verificar branco/cinza PRIMEIRO (cores neutras com baixa saturação)
    # pois elas podem ter qualquer hue e não devem ser confundidas com outras cores
    
    total_valid = mask.sum()
    if total_valid < 10:
        return "-"
    
    min_pixels = max(10, int(total_valid * 0.2))  # Pelo menos 20% dos pixels
    
    # PRIORIDADE 1: Branco e Cinza (cores neutras - baixa saturação)
    # Branco: baixa saturação (< 40), alta luminosidade (> 180)
    white_mask = mask & (saturations_2d < 40) & (values_2d > 180)
    white_count = white_mask.sum()
    
    # Cinza: baixa saturação (< 40), luminosidade intermediária (80-200)
    gray_mask = mask & (saturations_2d < 40) & (values_2d >= 80) & (values_2d <= 200)
    gray_count = gray_mask.sum()
    
    # Se branco ou cinza têm muitos pixels, retorna imediatamente (prioridade)
    if white_count >= min_pixels:
        return "white"
    if gray_count >= min_pixels:
        return "gray"
    
    # PRIORIDADE 2: Cores saturadas (só verifica se não for branco/cinza)
    # Remove pixels de baixa saturação para análise de cores saturadas
    saturated_mask = mask & (saturations_2d >= 40)
    if saturated_mask.sum() < min_pixels:
        # Se não há pixels saturados suficientes, pode ser branco/cinza escuro
        if white_count + gray_count >= min_pixels:
            return "white" if white_count > gray_count else "gray"
        return "-"
    
    # Pega apenas pixels saturados para análise de cor (agora extrai 1D dos pixels válidos)
    hues_sat = hues_2d[saturated_mask]
    values_sat = values_2d[saturated_mask]
    
    color_counts = {}
    
    # Vermelho (duas faixas devido ao wrap-around em 0/180)
    red_mask = ((hues_sat >= 0) & (hues_sat <= 10)) | ((hues_sat >= 170) & (hues_sat <= 179))
    color_counts["red"] = red_mask.sum()
    
    # Amarelo (range mais restrito para evitar confusão com laranja)
    yellow_mask = (hues_sat >= 20) & (hues_sat <= 30)
    color_counts["yellow"] = yellow_mask.sum()
    
    # Azul (range mais restrito)
    blue_mask = (hues_sat >= 100) & (hues_sat <= 130)
    color_counts["blue"] = blue_mask.sum()
    
    # Marrom: H 10-20, S já filtrado (>= 40), V intermediário
    brown_mask = (hues_sat >= 10) & (hues_sat < 20) & (values_sat >= 50) & (values_sat <= 150)
    color_counts["brown"] = brown_mask.sum()
    
    # Determina cor dominante entre as cores saturadas
    best_color = "-"
    best_count = 0
    
    for color_name, count in color_counts.items():
        if count >= min_pixels and count > best_count:
            best_count = count
            best_color = color_name
    
    return best_color

def _eval_flags_from_frame(raw_frame: Dict[str, List[Tuple[float, List[float]]]],
                           pos: Dict[str, List[str]],
                           person_box: List[int],
                           head_ratio: float,
                           chest_min: float,
                           chest_max: float,
                           frame_bgr: Optional[np.ndarray] = None,
                           debug: bool = False):
    """
    Avalia flags de EPIs a partir das detecções do YOLO-World.
    Agora detecta apenas "helmet" genérico e usa análise de pixels para determinar a cor.
    """
    Z_HEAD  = (0.0, max(0.05, min(head_ratio, 0.6)))
    # Zona do peito ampliada para cobrir ombros até cintura (mais permissiva)
    Z_CHEST = (max(0.0, min(chest_min, chest_max) - 0.1), min(1.0, max(chest_min, chest_max) + 0.1))
    # Zona do torso (para avental e colete) - ainda mais ampla
    Z_TORSO = (0.20, 0.90)  # Cobre do pescoço até quase a cintura
    Z_HANDS = (0.50, 1.00)

    # Detecta apenas capacete genérico (sem cores específicas)
    helmet_present, helmet_score, helmet_box = _best_hit_in_group(raw_frame, pos.get("helmet", []), person_box, Z_HEAD)
    
    # Debug: mostra todas as detecções brutas de capacete
    if debug:
        helmet_detections = []
        for s in pos.get("helmet", []):
            for score, b in raw_frame.get(s, []):
                cx, cy = _center(b)
                if _inside(cx, cy, person_box):
                    helmet_detections.append((s, score, b, _in_vertical_zone(cy, person_box, Z_HEAD[0], Z_HEAD[1])))
        if helmet_detections:
            print(f"  [DEBUG EPI] Capacete - Detecções brutas: {len(helmet_detections)}")
            for syn, sc, box, in_zone in helmet_detections[:3]:  # Mostra até 3
                print(f"    '{syn}': score={sc:.3f}, zona={in_zone}, box={[int(x) for x in box]}")
    
    # Detecta cor do capacete via análise de pixels
    helmet_color = "-"
    helmet_detected_colors = {
        "helmet_red": False,
        "helmet_blue": False,
        "helmet_yellow": False,
        "helmet_white": False,
        "helmet_brown": False,
    }
    
    if helmet_present and helmet_box and frame_bgr is not None:
        detected_color = _detect_helmet_color_from_pixels(frame_bgr, helmet_box)
        if detected_color != "-":
            helmet_color = detected_color
            # Mapeia cor detectada para as flags
            color_map = {
                "red": "helmet_red",
                "blue": "helmet_blue",
                "yellow": "helmet_yellow",
                "white": "helmet_white",
                "gray": "helmet_white",  # Cinza é tratado como branco para EPI
                "brown": "helmet_brown",
            }
            if detected_color in color_map:
                helmet_detected_colors[color_map[detected_color]] = True

    gloves_present, gloves_score, _ = _best_hit_in_group(raw_frame, pos.get("gloves", []), person_box, Z_HANDS)
    ear_present, ear_score, _ = _best_hit_in_group(raw_frame, pos.get("ear_protection", []), person_box, Z_HEAD)
    
    # Colete refletivo / Roupa de proteção - SEM restrição de zona (None = aceita qualquer posição dentro do box da pessoa)
    # A roupa pode estar em qualquer parte do corpo (ombros, peito, cintura, etc.)
    vest_present, vest_score, vest_box = _best_hit_in_group(raw_frame, pos.get("vest", []), person_box, None)
    
    # Avental - também SEM restrição de zona
    apron_present, apron_score, apron_box = _best_hit_in_group(raw_frame, pos.get("apron", []), person_box, None)
    
    # Debug: mostra TODAS as detecções brutas de colete/roupa (mesmo fora da zona, se houver)
    if debug:
        vest_detections_all = []
        for s in pos.get("vest", []):
            for score, b in raw_frame.get(s, []):
                cx, cy = _center(b)
                if _inside(cx, cy, person_box):
                    # Calcula posição relativa para debug
                    person_h = person_box[3] - person_box[1]
                    rel_y = (cy - person_box[1]) / person_h if person_h > 0 else 0.5
                    vest_detections_all.append((s, score, b, rel_y))
        
        if vest_detections_all:
            print(f"  [DEBUG EPI] Colete/Roupa - {len(vest_detections_all)} detecção(ões) brutas encontradas:")
            for syn, sc, box, rel_y in vest_detections_all[:5]:  # Mostra até 5
                print(f"    '{syn}': score={sc:.3f}, pos_y_relativa={rel_y:.2f} (0=topo, 1=base), box={[int(x) for x in box]}")
            if vest_present:
                print(f"  [DEBUG EPI] ✅ Colete DETECTADO (melhor score: {vest_score:.3f})")
            else:
                print(f"  [DEBUG EPI] ❌ Colete NÃO aceito (melhor score: {max([s for _, s, _, _ in vest_detections_all], default=0):.3f})")
        elif pos.get("vest"):
            print(f"  [DEBUG EPI] ❌ Colete - NENHUMA detecção bruta encontrada para prompts: {pos.get('vest', [])}")
            print(f"  [DEBUG EPI]    Total de detecções brutas no frame: {sum(len(raw_frame.get(s, [])) for s in pos.get('vest', []))}")
    
    if debug:
        apron_detections_all = []
        for s in pos.get("apron", []):
            for score, b in raw_frame.get(s, []):
                cx, cy = _center(b)
                if _inside(cx, cy, person_box):
                    person_h = person_box[3] - person_box[1]
                    rel_y = (cy - person_box[1]) / person_h if person_h > 0 else 0.5
                    apron_detections_all.append((s, score, b, rel_y))
        
        if apron_detections_all:
            print(f"  [DEBUG EPI] Avental - {len(apron_detections_all)} detecção(ões) brutas encontradas:")
            for syn, sc, box, rel_y in apron_detections_all[:3]:
                print(f"    '{syn}': score={sc:.3f}, pos_y_relativa={rel_y:.2f}, box={[int(x) for x in box]}")
    
    # Se colete ou avental detectado, considera como "vest" (compatibilidade)
    # Mas também mantém "apron" separado
    vest_final = vest_present or apron_present

    flags = {
        "helmet": bool(helmet_present),
        "helmet_red": helmet_detected_colors["helmet_red"],
        "helmet_blue": helmet_detected_colors["helmet_blue"],
        "helmet_yellow": helmet_detected_colors["helmet_yellow"],
        "helmet_white": helmet_detected_colors["helmet_white"],
        "helmet_brown": helmet_detected_colors["helmet_brown"],
        "gloves": bool(gloves_present),
        "ear_protection": bool(ear_present),
        "vest": bool(vest_final),  # Inclui colete OU avental
        "apron": bool(apron_present),  # Avental separado
    }
    
    # Retorna cor formatada para exibição (mantém formato original para compatibilidade)
    # A tradução será feita no draw_ppe_panel
    color_display = f"{helmet_color} helmet" if helmet_color != "-" else "-"
    return flags, color_display

# --------------------- ROI utils ---------------------
def _point_in_poly(pt: Tuple[float, float], poly_pts: List[Tuple[int,int]]) -> bool:
    x, y = pt
    inside = False
    n = len(poly_pts)
    for i in range(n):
        x1, y1 = poly_pts[i]
        x2, y2 = poly_pts[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y))
        if cond:
            xin = (x2 - x1) * (y - y1) / ( (y2 - y1) + 1e-12 ) + x1
            if x < xin:
                inside = not inside
    return inside

class ROI:
    def __init__(self, json_path: Optional[str], use_polygons: Optional[List[str]] = None, debug: bool = False):
        self.polygons = []
        self.lines = []
        self.debug = debug
        self.expected_resolution = None  # Resolução do vídeo para o qual o ROI foi definido
        self.original_polygons = []  # Armazena polígonos originais para escalonamento
        self.original_lines = []  # Armazena linhas originais para escalonamento
        self.scaled = False  # Flag para indicar se já foi escalado
        self.original_video_resolution = None  # Resolução original do vídeo (detectada no primeiro frame)
        
        if json_path:
            if not os.path.isabs(json_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                json_path = os.path.join(script_dir, json_path)
            
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Arquivo ROI não encontrado: {json_path}")
            
            with open(json_path, "r") as f:
                data = json.load(f)
            
            available_polygons = [poly.get("name", "") for poly in data.get("polygons", [])]
            
            if use_polygons is not None:
                invalid_names = [name for name in use_polygons if name not in available_polygons]
                if invalid_names:
                    print(f"[WARN] ROI: Polígonos não encontrados no JSON: {invalid_names}")
                    print(f"[INFO] ROI: Polígonos disponíveis: {available_polygons}")
                valid_names = [name for name in use_polygons if name in available_polygons]
            else:
                valid_names = available_polygons
            
            loaded_names = []
            for poly in data.get("polygons", []):
                name = poly.get("name", "")
                if use_polygons is not None and name not in valid_names:
                    continue
                pts = [(int(p[0]), int(p[1])) for p in poly.get("points", [])]
                if len(pts) >= 3:
                    # Armazena polígono original e processado
                    self.original_polygons.append({"name": name, "pts": pts})
                    self.polygons.append({"name": name, "pts": pts})
                    loaded_names.append(name)
                    # Calcula resolução esperada baseada no maior ponto do ROI
                    # Isso será usado como referência inicial, mas será atualizado com a resolução real do primeiro frame
                    # Não calcula expected_resolution aqui - será detectado no primeiro frame
                    # O expected_resolution será definido quando validate_resolution for chamado pela primeira vez
            
            if loaded_names:
                print(f"[INFO] ROI: Carregados {len(loaded_names)} polígono(s): {loaded_names}")
                # Resolução será detectada no primeiro frame
            else:
                print(f"[WARN] ROI: Nenhum polígono carregado!")
            
            for ln in data.get("lines", []):
                coords = ln.get("coords", [])
                if len(coords) >= 2:
                    p1 = (int(coords[0][0]), int(coords[0][1]))
                    p2 = (int(coords[1][0]), int(coords[1][1]))
                    line_data = {"name": ln.get("name",""), "p1": p1, "p2": p2, "dir": int(ln.get("dir", 0))}
                    self.original_lines.append(line_data.copy())  # Armazena original
                    self.lines.append(line_data)  # Cópia para uso

    @property
    def active(self) -> bool:
        return len(self.polygons) > 0

    def validate_resolution(self, frame_shape: Tuple[int, int]) -> bool:
        """Valida e escala ROI para a resolução do frame atual."""
        if not self.active:
            return True
        
        h, w = frame_shape[:2]
        
        # Calcula a resolução original do ROI (baseada nos pontos máximos)
        if self.expected_resolution is None:
            max_x = 0
            max_y = 0
            for poly in self.original_polygons:
                for pt in poly["pts"]:
                    max_x = max(max_x, pt[0])
                    max_y = max(max_y, pt[1])
            
            if max_x > 0 and max_y > 0:
                # Resolução para a qual o ROI foi definido (baseado nos pontos máximos)
                self.expected_resolution = (max_x + 10, max_y + 10)
            else:
                # Se não há pontos, assume que o ROI foi definido para a resolução atual
                self.expected_resolution = (w, h)
        
        # Sempre escala do ROI original para a resolução atual do frame
        roi_w, roi_h = self.expected_resolution
        scale_x = w / roi_w
        scale_y = h / roi_h
        
        # Verifica se já está na escala correta (tolerância de 0.1%)
        needs_rescale = True
        if self.scaled and self.polygons:
            # Verifica se a escala atual está próxima da necessária
            if self.original_polygons:
                orig_pt = self.original_polygons[0]["pts"][0]
                current_pt = self.polygons[0]["pts"][0]
                if orig_pt[0] > 0 and orig_pt[1] > 0:
                    current_scale_x = current_pt[0] / orig_pt[0]
                    current_scale_y = current_pt[1] / orig_pt[1]
                    
                    scale_diff_x = abs(current_scale_x - scale_x) / max(scale_x, 0.001)
                    scale_diff_y = abs(current_scale_y - scale_y) / max(scale_y, 0.001)
                    
                    if scale_diff_x < 0.001 and scale_diff_y < 0.001:
                        needs_rescale = False
        
        if needs_rescale:
            if not self.scaled or self.debug:
                print(f"[INFO] ROI: Escalando de {roi_w}x{roi_h} para {w}x{h} (escala: {scale_x:.4f}x, {scale_y:.4f}y)")
            
            # Escala os polígonos do ROI original para a resolução atual
            self.polygons = []
            for orig_poly in self.original_polygons:
                scaled_pts = [(int(p[0] * scale_x), int(p[1] * scale_y)) for p in orig_poly["pts"]]
                self.polygons.append({"name": orig_poly["name"], "pts": scaled_pts})
            
            # Escala as linhas
            self.lines = []
            for orig_ln in self.original_lines:
                scaled_p1 = (int(orig_ln["p1"][0] * scale_x), int(orig_ln["p1"][1] * scale_y))
                scaled_p2 = (int(orig_ln["p2"][0] * scale_x), int(orig_ln["p2"][1] * scale_y))
                self.lines.append({
                    "name": orig_ln["name"],
                    "p1": scaled_p1,
                    "p2": scaled_p2,
                    "dir": orig_ln["dir"]
                })
            
            self.scaled = True
            
            if self.debug:
                print(f"[INFO] ROI: Polígonos escalados. {len(self.polygons)} polígono(s), {len(self.lines)} linha(s)")
                # Debug: mostra alguns pontos antes e depois
                if self.original_polygons and self.polygons:
                    orig_pt = self.original_polygons[0]["pts"][0]
                    scaled_pt = self.polygons[0]["pts"][0]
                    print(f"[DEBUG] ROI: Ponto exemplo - Original: {orig_pt} -> Escalado: {scaled_pt}")
        
        return True

    def contains(self, pt: Tuple[float, float]) -> bool:
        if not self.active:
            return True
        return any(_point_in_poly(pt, poly["pts"]) for poly in self.polygons)

    def contains_box(self, box: List[int], min_overlap_ratio: float = 0.3) -> bool:
        if not self.active:
            return True
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        if box_area <= 0:
            return False
        samples_x = max(3, int((x2 - x1) / 20))
        samples_y = max(3, int((y2 - y1) / 20))
        inside_count = 0
        total_samples = 0
        for i in range(samples_x):
            for j in range(samples_y):
                px = x1 + (x2 - x1) * i / max(samples_x - 1, 1)
                py = y1 + (y2 - y1) * j / max(samples_y - 1, 1)
                if self.contains((px, py)):
                    inside_count += 1
                total_samples += 1
        overlap_ratio = inside_count / max(total_samples, 1)
        return overlap_ratio >= min_overlap_ratio

    def which(self, pt: Tuple[float, float]) -> List[str]:
        if not self.active:
            return []
        names = []
        for poly in self.polygons:
            if _point_in_poly(pt, poly["pts"]):
                nm = poly.get("name") or ""
                if nm:
                    names.append(nm)
        return names

# --------------------- câmera + IO + métricas ---------------------
class Camera:
    def __init__(self, cam_id: str, uri: str, target_fps: float, debug: bool=False):
        self.id = cam_id
        self.uri = self._with_tcp_and_timeouts(uri)
        self.period = 1.0 / max(target_fps, 1e-3)
        self.next_ts = time.time()
        self.last_frame = None
        self.writer = None
        self.out_dir = None
        self.csv = None
        self.debug = debug
        self.cap = None
        self._open_capture()

    def _with_tcp_and_timeouts(self, uri: str) -> str:
        if uri.startswith("rtsp://"):
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                "rtsp_transport;tcp|stimeout;5000000|rw_timeout;5000000|max_delay;5000000"
            )
            sep = "&" if "?" in uri else "?"
            if "rtsp_transport=" not in uri:
                uri = f"{uri}{sep}rtsp_transport=tcp"
        return uri

    def _open_capture(self, retries:int=4, wait_sec:float=0.8):
        backoff = wait_sec
        for i in range(1, retries+1):
            self.cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG)
            if self.cap.isOpened():
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if self.debug: print(f"[{self.id}] RTSP aberto (tentativa {i}/{retries}).")
                t0 = time.time()
                for _ in range(6):
                    ok, _frm = self.cap.read()
                    if ok:
                        break
                    if time.time() - t0 > 2.5:
                        break
                return
            if self.debug: print(f"[{self.id}] Falha ao abrir RTSP (tentativa {i}/{retries}). Aguardando {backoff:.1f}s...")
            time.sleep(backoff)
            backoff = min(backoff * 1.6, 3.5)
        print(f"[WARN] {self.id}: não abriu stream após {retries} tentativas: {self.uri}")

    def should_grab(self):
        return time.time() >= self.next_ts

    def grab(self):
        if self.cap is None or not self.cap.isOpened():
            if self.debug: print(f"[{self.id}] Capture fechado. Reabrindo...")
            self._open_capture()

        ok, frame = (False, None)
        if self.cap is not None:
            t0 = time.time()
            tries = 0
            while time.time() - t0 < 1.2 and tries < 5:
                ok, frame = self.cap.read()
                tries += 1
                if ok and frame is not None:
                    break
                time.sleep(0.04)

        if not ok or frame is None:
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass
            time.sleep(0.4)
            self._open_capture()
            ok, frame = (False, None)
            if self.cap is not None:
                ok, frame = self.cap.read()

        if ok and frame is not None:
            self.last_frame = frame
            self.next_ts = time.time() + self.period
            if self.debug:
                print(f"[{self.id}] Frame OK: shape={frame.shape}")
        else:
            if self.debug:
                print(f"[{self.id}] Grab falhou. ok={ok}, frame=None")

        return ok, frame

    def ensure_writer(self, out_dir: str, fps: int, frame_shape):
        if self.writer is None:
            os.makedirs(out_dir, exist_ok=True)
            h, w = frame_shape[:2]
            # Usa 'avc1' (H.264) que finaliza corretamente o MP4
            # Alternativas: 'XVID' (AVI), 'MJPG' (AVI), 'avc1' (MP4 H.264)
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            video_path = os.path.join(out_dir, f"{self.id}.mp4")
            self.writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            self.out_dir = out_dir
            if self.debug:
                print(f"[{self.id}] VideoWriter aberto? {self.writer.isOpened()} -> {video_path}")
            if not self.writer.isOpened():
                # Fallback para XVID se avc1 não funcionar
                print(f"[{self.id}] avc1 falhou, tentando XVID...")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_path = os.path.join(out_dir, f"{self.id}.avi")
                self.writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
                if self.debug:
                    print(f"[{self.id}] VideoWriter (XVID) aberto? {self.writer.isOpened()} -> {video_path}")

    def ensure_metrics_csv(self, out_dir: str):
        if self.csv is None:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"metrics_{self.id}.csv")
            is_new = not os.path.exists(path)
            self.csv = open(path, "a", newline="")
            w = csv.writer(self.csv)
            if is_new:
                w.writerow(["ts","cam_id","persons","tracks","ppe_calls",
                            "rtdetr_ms","track_ms","match_ms","ppe_ms",
                            "agg_ms","draw_ms","io_ms","total_ms"])

    def close(self):
        """Fecha VideoWriter e VideoCapture corretamente."""
        if self.writer is not None:
            try:
                # Garante que o VideoWriter finalize corretamente o arquivo
                if self.writer.isOpened():
                    # Escreve um frame vazio para forçar finalização (se necessário)
                    # Mas na verdade, apenas release() deve ser suficiente
                    self.writer.release()
                self.writer = None
                if self.debug:
                    print(f"[{self.id}] VideoWriter fechado corretamente")
            except Exception as e:
                print(f"[{self.id}] Erro ao fechar VideoWriter: {e}")
                import traceback
                traceback.print_exc()
        
        if self.cap is not None:
            try:
                self.cap.release()
                self.cap = None
            except Exception as e:
                print(f"[{self.id}] Erro ao fechar VideoCapture: {e}")
        
        if self.csv is not None:
            try:
                self.csv.close()
                self.csv = None
            except Exception as e:
                print(f"[{self.id}] Erro ao fechar CSV: {e}")
    
    def __del__(self):
        """Destrutor - garante que recursos sejam liberados."""
        self.close()

    def write(self, frame, force_jpg: bool=False):
        if self.out_dir is None:
            return
        if force_jpg:
            cv2.imwrite(os.path.join(self.out_dir, f"{self.id}_latest.jpg"), frame)
        if self.writer is not None and self.writer.isOpened():
            self.writer.write(frame)
        else:
            cv2.imwrite(os.path.join(self.out_dir, f"{self.id}_frame_{int(time.time()*1000)}.jpg"), frame)

    def log_metrics(self, out_dir: str, row: List):
        if self.csv is None:
            self.ensure_metrics_csv(out_dir)
        csv.writer(self.csv).writerow(row)
        try:
            self.csv.flush()
        except Exception:
            pass

# --------------------------- main loop ---------------------------
def run(config_path="config/stream_rtdetr.yaml",
        prompt_path="config/ppe_prompts_rtdetr.yaml",
        roi_path: Optional[str] = None,
        roi_polys: Optional[List[str]] = None,
        draw_roi: bool = False,
        required_ppe: Optional[List[str]] = None,
        debug: bool = False,
        show_video: bool = False,
        save_video: Optional[bool] = None,
        show_rtdetr_boxes: bool = False,
        enable_alerts: bool = False,
        show_alert_grid: bool = False,
        alert_config_path: str = "db_config.env"):

    cfg = load_yaml(config_path)
    cams = [Camera(c["id"], c["uri"], cfg.get("target_fps", 0.5), debug=debug) for c in cfg["cameras"]]

    # RT-DETR-X para detecção de pessoas (com fallback para YOLOv8)
    person_detector = None
    try:
        rtdetr = RTDETRPerson(
            weights=cfg.get("rtdetr_weights", "rtdetr-x.pt"),
            device=cfg.get("device", "cuda"),
            imgsz=cfg.get("rtdetr_imgsz", 1280),
            conf=cfg.get("rtdetr_conf", 0.15),
            iou=cfg.get("rtdetr_iou", 0.45),
            min_area=float(cfg.get("rtdetr_min_area", 0.0001)),
            max_area=float(cfg.get("rtdetr_max_area", 0.8)),
            min_aspect_ratio=float(cfg.get("rtdetr_min_aspect_ratio", 0.25)),
            max_aspect_ratio=float(cfg.get("rtdetr_max_aspect_ratio", 6.0)),
            min_height_px=int(cfg.get("rtdetr_min_height_px", 20)),
            min_width_px=int(cfg.get("rtdetr_min_width_px", 10)),
            disable_filters=bool(cfg.get("rtdetr_disable_filters", False)),
            debug=debug
        )
        person_detector = rtdetr
        print("[INFO] RT-DETR-X carregado com sucesso")
    except Exception as e:
        print(f"[WARN] Falha ao carregar RT-DETR-X: {e}")
        print("[INFO] Usando YOLOv8 como fallback...")
        # Fallback para YOLOv8
        try:
            # Tenta importar do projeto original
            yolo_detector_path = os.path.join(egtc_olm_dir, 'yolo_detector.py')
            if os.path.exists(yolo_detector_path):
                sys.path.insert(0, egtc_olm_dir)
                from yolo_detector import YOLOPerson
            else:
                # Se não encontrar, cria uma versão simples usando Ultralytics
                from ultralytics import YOLO as YOLO_ULTRALYTICS
                class YOLOPerson:
                    def __init__(self, weights, imgsz=640, conf=0.25, iou=0.45):
                        self.model = YOLO_ULTRALYTICS(weights)
                        self.imgsz = imgsz
                        self.conf = conf
                        self.iou = iou
                    def detect(self, frame, debug=False):
                        results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, 
                                                     iou=self.iou, classes=[0], verbose=False)
                        boxes = []
                        if len(results) > 0 and results[0].boxes is not None:
                            for b in results[0].boxes:
                                xyxy = b.xyxy[0].cpu().numpy()
                                conf = float(b.conf[0].item())
                                boxes.append((float(xyxy[0]), float(xyxy[1]), 
                                            float(xyxy[2]), float(xyxy[3]), conf))
                        return boxes
            
            person_detector = YOLOPerson(
                weights=cfg.get("yolo_weights", "yolov8m.pt"),
                imgsz=cfg.get("yolo_imgsz", 1280),
                conf=cfg.get("yolo_conf", 0.25),
                iou=cfg.get("yolo_iou", 0.45),
            )
            print("[INFO] YOLOv8 carregado como fallback")
        except Exception as e2:
            print(f"[ERRO] Falha ao carregar YOLOv8: {e2}")
            raise

    # Detector de EPIs unificado (YOLO-World ou OWL-V2)
    ppe_detector_type = cfg.get("ppe_detector", "yolo-world").lower()  # "yolo-world" ou "owl-v2"
    
    # Prepara parâmetros para o detector unificado
    detector_kwargs = {
        "device": cfg.get("device", "cuda"),
    }
    
    if ppe_detector_type == "yolo-world":
        # Parâmetros YOLO-World
        detector_kwargs.update({
            "yw_model": cfg.get("yw_model", "yolov8m-world.pt"),
            "yw_fp16": bool(cfg.get("yw_fp16", True)),
            "yw_use_crop": bool(cfg.get("yw_use_crop", False)),
            "yw_crop_padding": float(cfg.get("yw_crop_padding", 0.20)),
            "yw_min_crop_size": int(cfg.get("yw_min_crop_size", 32)),
            "yw_imgsz": int(cfg.get("yw_imgsz", 1280)),
        })
    elif ppe_detector_type == "owl-v2":
        # Parâmetros OWL-V2
        detector_kwargs.update({
            "ovd_model": cfg.get("ovd_model", "google/owlv2-base-patch16"),
            "ovd_fp16": bool(cfg.get("ovd_fp16", True)),
            "ovd_cache_dir": cfg.get("ovd_cache_dir", "./.hf"),
            "ovd_use_fast": bool(cfg.get("ovd_use_fast", True)),
            "ovd_quantization_mode": cfg.get("ovd_quantization_mode", "none"),
        })
    else:
        raise ValueError(f"Tipo de detector PPE não suportado: {ppe_detector_type}. Use 'yolo-world' ou 'owl-v2'")
    
    ppe_detector = UnifiedPPEDetector(detector_type=ppe_detector_type, **detector_kwargs)

    pos, neg = read_prompts(prompt_path)

    # Saída
    out_dir = cfg.get("out_dir", "./out")
    os.makedirs(out_dir, exist_ok=True)
    if save_video is None:
        save_video = bool(cfg.get("save_video", True))
    video_fps = int(cfg.get("video_fps", 2))

    # Threshold para detecção de EPIs (YOLO-World ou OWL-V2)
    if ppe_detector_type == "yolo-world":
        ppe_score_thr = float(cfg.get("yw_score_thr", 0.15))
    else:  # owl-v2
        ppe_score_thr = float(cfg.get("ovd_score_thr", 0.26))

    # Tracking + debounce
    debounce_sec = float(cfg.get("debounce_seconds", 8.0))
    track_thresh = float(cfg.get("track_thresh", 0.25))
    match_thresh = float(cfg.get("match_thresh", 0.3))  # Reduzido para melhor persistência
    track_buffer = int(cfg.get("track_buffer", 60))  # Aumentado para manter tracks por mais tempo
    iou_thresh = float(cfg.get("track_iou_thresh", 0.3))  # Para SimpleIoUTracker
    max_age = int(cfg.get("track_max_age", 30))  # Para SimpleIoUTracker
    
    trackers = {
        cam.id: PPETracker(
            fps_hint=cfg.get("target_fps", 0.5),
            debounce_seconds=debounce_sec,
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            iou_thresh=iou_thresh,
            max_age=max_age
        ) for cam in cams
    }

    # Métricas
    metrics_overlay = bool(cfg.get("metrics_overlay", True))
    metrics_csv = bool(cfg.get("metrics_csv", True))
    metrics_print_every = int(cfg.get("metrics_print_every", 30))
    frame_counters = {cam.id: 0 for cam in cams}

    # Zonas verticais
    head_ratio = float(cfg.get("head_ratio", 0.45))
    chest_min = float(cfg.get("chest_min_ratio", 0.35))
    chest_max = float(cfg.get("chest_max_ratio", 0.75))

    # ROI
    roi = ROI(roi_path, use_polygons=roi_polys, debug=debug) if roi_path else ROI(None, debug=debug)
    
    # Aviso se ROI está carregado mas não será desenhado
    if roi.active and not draw_roi:
        print(f"[INFO] ROI carregado ({len(roi.polygons)} polígono(s)) mas não será desenhado. Use --draw-roi para visualizar.")
    
    # Configuração de EPIs obrigatórios por ROI
    roi_ppe_config = cfg.get("roi_ppe_config", {})  # {roi_name: [list of required EPIs]}
    if debug and roi_ppe_config:
        print(f"[INFO] Configuração de EPIs por ROI: {roi_ppe_config}")
    
    # EPIs obrigatórios padrão (usado quando não há ROI ou ROI não está na config)
    req_ppe_default = required_ppe or cfg.get("required_ppe", ["helmet", "gloves", "ear_protection", "vest"])

    # Sistema de alertas (pode ser habilitado via YAML ou CLI)
    enable_alerts = enable_alerts or bool(cfg.get("enable_alerts", False))
    show_alert_grid = show_alert_grid or bool(cfg.get("show_alert_grid", False))
    alert_managers = {}
    
    # Inicializa AlertManager se show_alert_grid OU enable_alerts estiver habilitado
    # (show_alert_grid controla exibição, enable_alerts controla envio)
    if show_alert_grid or enable_alerts:
        # Carrega configuração de alertas
        alert_config = AlertConfig.from_env_file(alert_config_path)
        # Ajusta configurações do YAML se disponíveis
        alert_config.alert_debounce_seconds = float(cfg.get("alert_debounce_seconds", alert_config.alert_debounce_seconds))
        alert_config.alert_min_consecutive_frames = int(cfg.get("alert_min_consecutive_frames", alert_config.alert_min_consecutive_frames))
        alert_config.suppression_reset_seconds = float(cfg.get("alert_suppression_reset_seconds", alert_config.suppression_reset_seconds))
        alert_config.alert_hash_ttl_seconds = float(cfg.get("alert_hash_ttl_seconds", alert_config.alert_hash_ttl_seconds))
        alert_config.grid_size = int(cfg.get("alert_grid_size", alert_config.grid_size))
        alert_config.timezone_offset_hours = float(cfg.get("timezone_offset_hours", alert_config.timezone_offset_hours))
        alert_config.save_alert_images = bool(cfg.get("save_alert_images", alert_config.save_alert_images))
        alert_config.save_crop_only = bool(cfg.get("save_crop_only", alert_config.save_crop_only))
        if "crops_dir" in cfg:
            alert_config.crops_dir = str(cfg["crops_dir"])
        
        # Inicializa AlertManager para cada câmera (será atualizado com resolução real no primeiro frame)
        for cam in cams:
            # Usa resolução padrão temporária, será atualizada no primeiro frame
            alert_managers[cam.id] = None
        
        if enable_alerts:
            print(f"[INFO] Sistema de alertas habilitado (envio ativo). Debounce: {alert_config.alert_debounce_seconds}s, Supressão: {alert_config.suppression_reset_seconds}s")
        else:
            print(f"[INFO] Sistema de alertas habilitado (apenas visualização, sem envio). Debounce: {alert_config.alert_debounce_seconds}s")

    print(f"[INFO] Pipeline RT-DETR-X + {ppe_detector_type.upper()} iniciado. roi_active={roi.active}. Debug={debug}.")
    print(f"[INFO] Exibição em tempo real: {'ON' if show_video else 'OFF'}")
    print(f"[INFO] Gravação de vídeo: {'ON' if save_video else 'OFF'}")
    print(f"[INFO] Boxes RT-DETR-X (laranja): {'ON' if show_rtdetr_boxes else 'OFF'}")
    print(f"[INFO] Sistema de alertas: {'ON' if enable_alerts else 'OFF'}")
    print(f"[INFO] Grid de alertas (visualização): {'ON' if show_alert_grid else 'OFF'}")
    try:
        while True:
            progressed = False
            for cam in cams:
                try:
                    if not cam.should_grab():
                        continue

                    t_total0 = time.perf_counter()
                    ok, frame = cam.grab()
                    if not ok or frame is None:
                        if debug: print(f"[{cam.id}] Grab falhou. ok={ok}, frame=None")
                        continue
                    progressed = True

                    H, W = frame.shape[:2]
                    frame_counters[cam.id] += 1

                    # Validação de resolução ROI (sempre valida para garantir escala correta)
                    if roi.active:
                        roi.validate_resolution(frame.shape)

                    # ==== RT-DETR-X (detecção de pessoas) ====
                    t1 = time.perf_counter()
                    boxes = person_detector.detect(frame, debug=debug)
                    t2 = time.perf_counter(); rtdetr_ms = (t2 - t1) * 1000.0
                    detector_name = "RT-DETR-X" if isinstance(person_detector, RTDETRPerson) else "YOLOv8"
                    if debug:
                        print(f"[{cam.id}] {detector_name}: {len(boxes)} pessoa(s) detectada(s) (antes do filtro ROI)")
                    
                    # ==== Filtro ROI: Apenas detecta pessoas dentro do ROI ====
                    if roi.active:
                        boxes_before_roi = len(boxes)
                        boxes = [box for box in boxes if roi.contains_box([int(box[0]), int(box[1]), int(box[2]), int(box[3])], min_overlap_ratio=0.3)]
                        if debug and boxes_before_roi != len(boxes):
                            print(f"[{cam.id}] Filtro ROI: {boxes_before_roi} -> {len(boxes)} pessoa(s) (dentro do ROI)")
                    elif len(boxes) == 0:
                        print(f"[WARN] {cam.id}: Nenhuma pessoa detectada no frame {frame_counters[cam.id]}")
                    
                    # Salva boxes brutos para debug (antes do tracking)
                    raw_boxes_for_debug = list(boxes) if boxes else []

                    # ==== TRACKER ====
                    t3 = time.perf_counter()
                    dets = np.array([[x1,y1,x2,y2,conf] for (x1,y1,x2,y2,conf) in boxes], dtype=float) if boxes else np.zeros((0,5), dtype=float)
                    tracks = trackers[cam.id].update(dets, frame_size=frame.shape[:2])
                    t4 = time.perf_counter(); track_ms = (t4 - t3) * 1000.0

                    persons = len(boxes); n_tracks = len(tracks)

                    # ==== Detecção de EPIs (YOLO-World ou OWL-V2) ====
                    ppe_ms = 0.0; agg_ms = 0.0
                    t7 = time.perf_counter()
                    
                    # Prepara boxes de pessoas para modo crop (apenas YOLO-World)
                    person_boxes = [[int(x1), int(y1), int(x2), int(y2)] for (x1, y1, x2, y2, _) in boxes] if boxes else []
                    
                    # Inferência unificada (YOLO-World ou OWL-V2)
                    _, raw_frame = ppe_detector.infer(
                        frame, 
                        pos, 
                        score_thr=ppe_score_thr, 
                        person_boxes=person_boxes if ppe_detector.use_crop else None,
                        negative=neg
                    )
                    
                    t8 = time.perf_counter(); ppe_ms = (t8 - t7) * 1000.0

                    # ==== ROI overlay ====
                    if draw_roi:
                        if roi.active:
                            if debug and frame_counters[cam.id] % 30 == 0:
                                print(f"[DEBUG] {cam.id}: Desenhando {len(roi.polygons)} polígono(s) e {len(roi.lines)} linha(s) do ROI")
                                for poly in roi.polygons:
                                    pts = poly.get("pts", [])
                                    if pts:
                                        min_x = min(p[0] for p in pts)
                                        max_x = max(p[0] for p in pts)
                                        min_y = min(p[1] for p in pts)
                                        max_y = max(p[1] for p in pts)
                                        print(f"  [DEBUG] Polígono '{poly.get('name')}': {len(pts)} pontos, bbox=({min_x},{min_y})-({max_x},{max_y})")
                            draw_rois(frame, roi.polygons, roi.lines, debug=debug)
                        elif debug and frame_counters[cam.id] % 30 == 0:
                            print(f"[DEBUG] {cam.id}: draw_roi=True mas ROI não está ativo (sem polígonos)")

                    # ==== Atribuição por track + debounce + ROI gating ====
                    t9 = time.perf_counter()
                    people_normal = []
                    people_violators = []
                    for tr in tracks:
                        tid, pbox = tr["id"], list(map(int, tr["box"]))
                        cx, cy = (0.5*(pbox[0]+pbox[2]), 0.5*(pbox[1]+pbox[3]))
                        inside = roi.contains_box(pbox, min_overlap_ratio=0.3) if roi.active else True
                        roi_names = roi.which((cx, cy)) if roi.active else []
                        roi_name_txt = ", ".join(roi_names) if roi_names else None

                        if inside:
                            flags, helmet_color = _eval_flags_from_frame(raw_frame, pos, pbox, head_ratio, chest_min, chest_max, frame_bgr=frame, debug=debug)
                            debounced = trackers[cam.id].update_ppe(tid, flags)
                            
                            # Determina EPIs obrigatórios baseado no ROI
                            # IMPORTANTE: Se ROI está ativo, só usa EPIs configurados no ROI
                            # Se não encontrar o ROI na config, usa lista vazia (não requer EPIs)
                            req_ppe = []
                            
                            if roi.active:
                                # ROI está ativo: tenta encontrar config específica do ROI
                                if roi_names and roi_ppe_config:
                                    # Procura o primeiro ROI que está na config
                                    for roi_name in roi_names:
                                        if roi_name in roi_ppe_config:
                                            req_ppe = roi_ppe_config[roi_name]
                                            if debug and frame_counters[cam.id] % 30 == 0:
                                                print(f"[DEBUG] {cam.id} ID {tid}: ROI '{roi_name}' encontrado na config. EPIs obrigatórios: {req_ppe}")
                                            break
                                    
                                    # Se não encontrou nenhum ROI na config, usa lista vazia
                                    if not req_ppe:
                                        if debug and frame_counters[cam.id] % 30 == 0:
                                            print(f"[DEBUG] {cam.id} ID {tid}: ROI(s) {roi_names} não encontrado(s) na config. Nenhum EPI obrigatório.")
                                else:
                                    # ROI ativo mas sem config ou sem nomes: usa lista vazia
                                    if debug and frame_counters[cam.id] % 30 == 0:
                                        print(f"[DEBUG] {cam.id} ID {tid}: ROI ativo mas sem config ou sem nomes. Nenhum EPI obrigatório.")
                            else:
                                # ROI não está ativo: usa EPIs padrão
                                req_ppe = req_ppe_default
                                if debug and frame_counters[cam.id] % 30 == 0:
                                    print(f"[DEBUG] {cam.id} ID {tid}: ROI não ativo. Usando EPIs padrão: {req_ppe}")
                            
                            # Verifica violações: se requer capacete específico (ex: helmet_white), verifica a cor
                            # Se requer apenas "helmet", aceita qualquer capacete
                            is_violation = False
                            violation_items = []  # Lista de EPIs que estão gerando violações
                            
                            if req_ppe:  # Só verifica violações se houver EPIs obrigatórios
                                for req_item in req_ppe:
                                    is_item_violation = False
                                    if req_item == "helmet":
                                        # Se requer apenas "helmet", aceita qualquer capacete
                                        if debounced.get("helmet") is False:
                                            is_violation = True
                                            is_item_violation = True
                                            if debug and frame_counters[cam.id] % 30 == 0:
                                                print(f"[DEBUG] {cam.id} ID {tid}: Violação - capacete ausente")
                                    elif req_item.startswith("helmet_"):
                                        # Se requer capacete de cor específica, verifica a cor
                                        if debounced.get(req_item) is False:
                                            is_violation = True
                                            is_item_violation = True
                                            if debug and frame_counters[cam.id] % 30 == 0:
                                                print(f"[DEBUG] {cam.id} ID {tid}: Violação - {req_item} ausente")
                                    else:
                                        # Outros EPIs (gloves, vest, etc.)
                                        if debounced.get(req_item) is False:
                                            is_violation = True
                                            is_item_violation = True
                                            if debug and frame_counters[cam.id] % 30 == 0:
                                                print(f"[DEBUG] {cam.id} ID {tid}: Violação - {req_item} ausente")
                                    
                                    # Adiciona à lista de violações se este item está gerando alarme
                                    if is_item_violation:
                                        violation_items.append(req_item)
                            
                            # Cria ppe_flags completo (para uso interno)
                            ppe_flags_full = {
                                "helmet": debounced.get("helmet", "PENDING"),
                                "gloves": debounced.get("gloves", "PENDING"),
                                "ear_protection": debounced.get("ear_protection", "PENDING"),
                                "vest": debounced.get("vest", "PENDING"),
                                "helmet_color": (helmet_color if debounced.get("helmet") is True else "-"),
                            }
                            
                            # Cria ppe_flags apenas com EPIs que estão sendo monitorados (req_ppe)
                            # Mostra todos os EPIs monitorados, independente de serem violações ou não
                            ppe_flags_display = {}
                            if req_ppe:  # Só cria painel se houver EPIs sendo monitorados
                                for item in req_ppe:
                                    if item == "helmet":
                                        ppe_flags_display["helmet"] = debounced.get("helmet", "PENDING")
                                        # Inclui cor do capacete se capacete estiver presente
                                        if debounced.get("helmet") is True:
                                            ppe_flags_display["helmet_color"] = (helmet_color if helmet_color != "-" else "-")
                                        else:
                                            ppe_flags_display["helmet_color"] = "-"
                                    elif item.startswith("helmet_"):
                                        # Para capacete de cor específica, mostra o capacete genérico e a cor específica
                                        ppe_flags_display["helmet"] = debounced.get("helmet", "PENDING")
                                        ppe_flags_display[item] = debounced.get(item, "PENDING")
                                        if debounced.get("helmet") is True:
                                            ppe_flags_display["helmet_color"] = (helmet_color if helmet_color != "-" else "-")
                                        else:
                                            ppe_flags_display["helmet_color"] = "-"
                                    else:
                                        # Outros EPIs (gloves, vest, etc.)
                                        ppe_flags_display[item] = debounced.get(item, "PENDING")
                            
                            if is_violation:
                                # Para violadores: passa os EPIs monitorados (mostra todos, incluindo os que estão OK)
                                people_violators.append((tid, pbox, ppe_flags_display, roi_name_txt))
                            else:
                                # Para pessoas normais: também mostra os EPIs monitorados (todos estarão OK)
                                people_normal.append((tid, pbox, ppe_flags_display, roi_name_txt))
                        else:
                            people_normal.append((tid, pbox, None, roi_name_txt))
                    t10 = time.perf_counter(); agg_ms = (t10 - t9) * 1000.0

                    # ==== Sistema de Alertas ====
                    # Processa alertas se show_alert_grid OU enable_alerts estiver habilitado
                    if show_alert_grid or enable_alerts:
                        # Inicializa AlertManager no primeiro frame com resolução real
                        if alert_managers.get(cam.id) is None:
                            H, W = frame.shape[:2]
                            alert_config = AlertConfig.from_env_file(alert_config_path)
                            alert_config.alert_debounce_seconds = float(cfg.get("alert_debounce_seconds", alert_config.alert_debounce_seconds))
                            alert_config.alert_min_consecutive_frames = int(cfg.get("alert_min_consecutive_frames", alert_config.alert_min_consecutive_frames))
                            alert_config.suppression_reset_seconds = float(cfg.get("alert_suppression_reset_seconds", alert_config.suppression_reset_seconds))
                            alert_config.alert_hash_ttl_seconds = float(cfg.get("alert_hash_ttl_seconds", alert_config.alert_hash_ttl_seconds))
                            alert_config.grid_size = int(cfg.get("alert_grid_size", alert_config.grid_size))
                            alert_config.timezone_offset_hours = float(cfg.get("timezone_offset_hours", alert_config.timezone_offset_hours))
                            alert_config.save_alert_images = bool(cfg.get("save_alert_images", alert_config.save_alert_images))
                            alert_config.save_crop_only = bool(cfg.get("save_crop_only", alert_config.save_crop_only))
                            if "crops_dir" in cfg:
                                alert_config.crops_dir = str(cfg["crops_dir"])
                            alert_managers[cam.id] = AlertManager(alert_config, cam.id, W, H, send_alerts=enable_alerts)
                            print(f"[INFO] {cam.id}: AlertManager inicializado com resolução {W}x{H}, envio={'ON' if enable_alerts else 'OFF'}")
                        
                        alert_manager = alert_managers[cam.id]
                        
                        # Coleta violações para atualizar estado
                        violations_for_alert = []
                        for tid, pbox, ppe_flags, roi_name_txt in people_violators:
                            # Extrai EPIs faltando
                            missing_ppe = []
                            if ppe_flags:
                                for epi_key, epi_status in ppe_flags.items():
                                    if epi_key != "helmet_color" and epi_status is False:
                                        missing_ppe.append(epi_key)
                            if missing_ppe:
                                # Inclui ROI na tupla de violação
                                if debug:
                                    print(f"[DEBUG] {cam.id}: Violação para pessoa {tid} - ROI: {roi_name_txt}, EPIs faltando: {missing_ppe}")
                                violations_for_alert.append((tid, pbox, missing_ppe, roi_name_txt))
                        
                        # Atualiza estado de violações
                        if violations_for_alert:
                            alert_manager.update_violations(violations_for_alert)
                        
                        # Verifica e gera alertas (passa frame para Telegram apenas se enable_alerts=True)
                        # A lógica sempre roda para atualizar o grid, mas só envia se enable_alerts=True
                        alerts_generated = alert_manager.check_and_generate_alerts(frame_bgr=frame if enable_alerts else None)
                        if alerts_generated and debug:
                            if enable_alerts:
                                print(f"[ALERT] {cam.id}: {len(alerts_generated)} alerta(s) gerado(s) e enviado(s)")
                            else:
                                print(f"[ALERT] {cam.id}: {len(alerts_generated)} alerta(s) detectado(s) (não enviado - enable_alerts=False)")
                    else:
                        # Sistema de alertas não está ativo
                        alerts_generated = []

                    # ==== desenho ====
                    if save_video:
                        cam.ensure_writer(out_dir, video_fps, frame.shape)
                        if cam.writer is None or not cam.writer.isOpened():
                            if debug:
                                print(f"[{cam.id}] VideoWriter indisponível. Fallback para JPG.")
                            os.makedirs(out_dir, exist_ok=True)

                    # ==== Desenha boxes laranjas das detecções brutas do RT-DETR-X ====
                    # (desenha primeiro, por baixo dos boxes finais, com offset de 5px para ficar visível)
                    if show_rtdetr_boxes and raw_boxes_for_debug:
                        if debug and frame_counters[cam.id] % 30 == 0:
                            print(f"[DEBUG] {cam.id}: Desenhando {len(raw_boxes_for_debug)} box(es) laranja(s) do RT-DETR-X")
                        for idx, (x1, y1, x2, y2, conf) in enumerate(raw_boxes_for_debug):
                            box_raw = [float(x1), float(y1), float(x2), float(y2)]
                            if debug and frame_counters[cam.id] % 30 == 0:
                                print(f"  [DEBUG] Box laranja {idx}: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f}), conf={conf:.3f}, cor={ORANGE}")
                            # Aplica offset de 5 pixels para que o box laranja fique visível mesmo quando os verdes/vermelhos são desenhados por cima
                            offset = 5
                            x1_i = max(0, int(x1) - offset)
                            y1_i = max(0, int(y1) - offset)
                            x2_i = min(W - 1, int(x2) + offset)
                            y2_i = min(H - 1, int(y2) + offset)
                            # Desenha box laranja diretamente com cv2 para garantir visibilidade
                            cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), ORANGE, 3, cv2.LINE_AA)
                            # Label simples
                            label_text = f"RT-DETR {conf:.2f}"  # Mantém RT-DETR (nome técnico)
                            # Remove acentos para renderização (OpenCV não suporta acentos)
                            from utils import _remove_accents
                            label_text_clean = _remove_accents(label_text)
                            (tw, th), _ = cv2.getTextSize(label_text_clean, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame, (x1_i, y1_i - th - 8), (x1_i + tw + 8, y1_i), ORANGE, -1, cv2.LINE_AA)
                            cv2.putText(frame, label_text_clean, (x1_i + 4, y1_i - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # ==== Desenha grid de alertas (se habilitado) ====
                    # Desenha antes dos boxes de pessoas para ficar por baixo
                    # show_alert_grid controla apenas a exibição visual, independente de enable_alerts
                    if show_alert_grid and alert_managers.get(cam.id):
                        alert_manager = alert_managers[cam.id]
                        suppressed_cells = alert_manager.get_suppressed_cells()
                        # Células com violações ativas (mas ainda não alertadas)
                        # Agora violation_states é indexado por track_id, então precisamos extrair as células
                        violation_cells = []
                        for track_id, state in alert_manager.violation_states.items():
                            if not state.get("alerted", False):  # Apenas violações não alertadas
                                violation_cells.append((state["grid_x"], state["grid_y"]))
                        if debug and frame_counters[cam.id] % 30 == 0:
                            print(f"[DEBUG GRID] {cam.id}: Desenhando grid - suprimidas: {len(suppressed_cells)}, violações: {len(violation_cells)}")
                        draw_alert_grid(frame, grid_size=alert_manager.config.grid_size, 
                                       suppressed_cells=suppressed_cells,
                                       violation_cells=violation_cells)
                    elif show_alert_grid and debug and frame_counters[cam.id] % 30 == 0:
                        # Debug: mostra por que não está desenhando
                        if not alert_managers.get(cam.id):
                            print(f"[DEBUG GRID] {cam.id}: Grid NÃO desenhado - AlertManager ainda não inicializado")
                    
                    # normais (pessoas em conformidade)
                    for (tid, box, ppe_flags, roi_name_txt) in people_normal:
                        draw_person_box(frame, box, color=GREEN, label=f"ID {tid}")
                        if ppe_flags:
                            px = min(int(box[0]) + 5, W - 260)
                            py = min(int(box[3]) + 5, H - 200)
                            # Verifica status do alerta (se sistema de alertas estiver ativo)
                            alert_status = None
                            if (show_alert_grid or enable_alerts) and alert_managers.get(cam.id):
                                alert_manager = alert_managers[cam.id]
                                cx = (box[0] + box[2]) / 2
                                cy = (box[1] + box[3]) / 2
                                alert_status = alert_manager.get_alert_status(cx, cy, track_id=tid)
                            draw_ppe_panel(frame, (px, py), ppe_flags, person_id=tid, roi_name=roi_name_txt, font_scale=0.55, alert_status=alert_status)
                    # violadores (desenha por último para ficar por cima)
                    for (tid, box, ppe_flags, roi_name_txt) in people_violators:
                        # Verifica status do alerta para determinar cor e texto do bounding box
                        box_color = RED  # Padrão: vermelho
                        box_label = f"VIOLAÇÃO ID {tid}"  # Padrão: VIOLAÇÃO
                        
                        if (show_alert_grid or enable_alerts) and alert_managers.get(cam.id):
                            alert_manager = alert_managers[cam.id]
                            cx = (box[0] + box[2]) / 2
                            cy = (box[1] + box[3]) / 2
                            alert_status = alert_manager.get_alert_status(cx, cy, track_id=tid)
                            
                            # Determina cor e texto do bounding box baseado no status
                            if alert_status == "ALERTA GERADO":
                                # Alerta foi enviado - vermelho com texto "ALERTA"
                                box_color = RED
                                box_label = f"ALERTA ID {tid}"
                            else:
                                # Violação detectada mas alerta ainda não enviado - amarelo com texto "AVALIANDO"
                                box_color = YELLOW
                                box_label = f"AVALIANDO ID {tid}"
                        else:
                            # Sistema de alertas não ativo - mantém padrão vermelho
                            pass
                        
                        # Desenha box com cor e texto apropriados
                        draw_person_box(frame, box, color=box_color, label=box_label)
                        
                        if ppe_flags:
                            px = min(int(box[0]) + 5, W - 260)
                            py = min(int(box[3]) + 5, H - 200)
                            # Verifica status do alerta para o painel (se sistema de alertas estiver ativo)
                            alert_status = None
                            if (show_alert_grid or enable_alerts) and alert_managers.get(cam.id):
                                alert_manager = alert_managers[cam.id]
                                cx = (box[0] + box[2]) / 2
                                cy = (box[1] + box[3]) / 2
                                alert_status = alert_manager.get_alert_status(cx, cy, track_id=tid)
                            draw_ppe_panel(frame, (px, py), ppe_flags, person_id=tid, roi_name=roi_name_txt, font_scale=0.55, alert_status=alert_status)

                    # banner (em português)
                    total_ms = (time.perf_counter() - t_total0) * 1000.0
                    
                    # Determina status de violação/alerta para o banner
                    violation_status = None
                    violation_color = None
                    
                    # Lógica: mostra "VIOLACAO" quando há violadores (aguardando confirmação)
                    # mostra "ALERTA" apenas quando alerta foi realmente enviado
                    # Prioriza mostrar violações pendentes (amarelo) para melhor visibilidade
                    if people_violators:
                        # Há violações detectadas
                        if alerts_generated and len(alerts_generated) > 0:
                            # Alerta foi enviado neste frame - mostra ALERTA em vermelho
                            violation_status = "ALERTA"
                            violation_color = RED  # Vermelho para alerta enviado
                        else:
                            # Há violações mas alerta ainda não foi enviado (aguardando confirmação)
                            violation_status = "VIOLACAO"
                            violation_color = YELLOW  # Amarelo para violação pendente
                    elif alerts_generated and len(alerts_generated) > 0:
                        # Alerta foi enviado mas não há violadores visíveis neste frame
                        violation_status = "ALERTA"
                        violation_color = RED  # Vermelho para alerta enviado
                    
                    if metrics_overlay:
                        banner = (f"{cam.id} | pessoas {persons} | rastros {len(tracks)} | "
                                  f"RT-DETR {rtdetr_ms:.0f}ms | rastr {track_ms:.0f}ms | "
                                  f"PPE {ppe_ms:.0f}ms | agg {agg_ms:.0f}ms | "
                                  f"ROI {'ligado' if roi.active else 'desligado'} | total {total_ms:.0f}ms")
                    else:
                        banner = f"{cam.id}  |  pessoas: {persons}  |  rastros: {len(tracks)}"
                    
                    # Adiciona status de violação/alerta ao banner se houver
                    if violation_status:
                        banner = f"{banner}  |  {violation_status}"
                    
                    # Desenha banner com cor apropriada
                    if violation_color:
                        put_banner(frame, banner, color=violation_color)
                    else:
                        put_banner(frame, banner)

                    # ==== Exibição em tempo real ====
                    if show_video:
                        window_name = f"Pipeline RT-DETR-X + {ppe_detector_type.upper()} - {cam.id}"
                        cv2.imshow(window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print(f"\n[INFO] Tecla 'q' pressionada. Encerrando...")
                            raise KeyboardInterrupt

                    # gravação / snapshots
                    if save_video:
                        cam.ensure_writer(out_dir, video_fps, frame.shape)
                        cam.write(frame, force_jpg=debug)
                    else:
                        if debug:
                            os.makedirs(out_dir, exist_ok=True)
                            cv2.imwrite(os.path.join(out_dir, f"{cam.id}_latest.jpg"), frame)
                    
                    # ==== MJPEG Streaming: Salva frame para streaming (sempre) ====
                    stream_dir = os.path.join(out_dir, "stream")
                    os.makedirs(stream_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(stream_dir, f"{cam.id}.jpg"), frame)

                    # ==== CSV ====
                    if metrics_csv:
                        cam.log_metrics(out_dir, [
                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            cam.id, persons, len(tracks), len(person_boxes) if ppe_detector.use_crop else 1,
                            f"{rtdetr_ms:.2f}", f"{track_ms:.2f}", "0.00",
                            f"{ppe_ms:.2f}", f"{agg_ms:.2f}", "0.00",
                            "0.00", f"{total_ms:.2f}"
                        ])

                    # ==== print periódico ====
                    if metrics_print_every > 0 and (frame_counters[cam.id] % metrics_print_every == 0):
                        print(f"[METRICS] {cam.id} | RT-DETR {rtdetr_ms:.1f} | trk {track_ms:.1f} | "
                              f"PPE {ppe_ms:.1f} | total {total_ms:.1f} | ROI={'on' if roi.active else 'off'}")

                except Exception as e:
                    print(f"[ERROR] {cam.id}: {e}")
                    traceback.print_exc()
                    try:
                        if cam.last_frame is not None:
                            os.makedirs(out_dir, exist_ok=True)
                            cv2.imwrite(os.path.join(out_dir, f"{cam.id}_error_last.jpg"), cam.last_frame)
                    except Exception:
                        pass

            if not progressed:
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[INFO] Encerrando...")
    finally:
        cv2.destroyAllWindows()
        for cam in cams:
            try:
                if cam.writer: cam.writer.release()
                if cam.cap: cam.cap.release()
                if cam.csv: cam.csv.close()
            except Exception:
                pass

# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EGTC pipeline: RT-DETR-X (pessoas) + YOLO-World (EPIs)")
    ap.add_argument("--config", default="config/stream_rtdetr.yaml")
    ap.add_argument("--prompts", default="config/ppe_prompts_rtdetr.yaml")
    ap.add_argument("--roi", default=None, help="Caminho do arquivo JSON de ROI")
    ap.add_argument("--roi-polys", default=None, help="Nomes de polígonos do ROI (separados por vírgula)")
    ap.add_argument("--draw-roi", action="store_true", help="Desenha os polígonos/linhas do ROI na saída")
    ap.add_argument("--required-ppe", default=None, help="EPIs obrigatórios, separados por vírgula (sobrepõe YAML)")
    ap.add_argument("--debug", action="store_true", help="Modo diagnóstico: logs e snapshots JPG")
    ap.add_argument("--show-video", action="store_true", help="Exibe vídeo de saída em tempo real (cv2.imshow)")
    ap.add_argument("--no-save-video", action="store_true", help="Desabilita gravação de vídeo (sobrescreve YAML)")
    ap.add_argument("--show-rtdetr-boxes", action="store_true", help="Exibe bounding boxes laranjas das detecções brutas do RT-DETR-X")
    ap.add_argument("--enable-alerts", action="store_true", help="Habilita sistema de alertas com persistência em banco")
    ap.add_argument("--show-alert-grid", action="store_true", help="Exibe grid 8x8 de alertas na tela (requer --enable-alerts)")
    ap.add_argument("--alert-config", default="db_config.env", help="Caminho do arquivo de configuração de banco/Redis")

    args = ap.parse_args()
    roi_polys = [s.strip() for s in args.roi_polys.split(",")] if args.roi_polys else None
    req_ppe = [s.strip() for s in args.required_ppe.split(",")] if args.required_ppe else None
    
    save_video = None if not args.no_save_video else False

    run(config_path=args.config,
        prompt_path=args.prompts,
        roi_path=args.roi,
        roi_polys=roi_polys,
        draw_roi=args.draw_roi,
        required_ppe=req_ppe,
        debug=args.debug,
        show_video=args.show_video,
        save_video=save_video,
        show_rtdetr_boxes=args.show_rtdetr_boxes,
        enable_alerts=args.enable_alerts,
        show_alert_grid=args.show_alert_grid,
        alert_config_path=args.alert_config)

