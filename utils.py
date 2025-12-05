# utils.py — UI fina + ID no painel + ROI no painel + desenho de ROI
import cv2
import numpy as np
from typing import Dict, Tuple, Union, Optional, List

# ---------------- cores (BGR) ----------------
GREEN  = (0, 255, 100)      # Verde mais vibrante
RED    = (0, 0, 255)        # Vermelho puro
YELLOW = (0, 255, 255)      # Amarelo puro
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GRAY   = (40, 40, 40)       # Mais escuro para melhor contraste
LIGHT  = (220, 220, 220)    # Mais claro
BLUE   = (255, 150, 0)      # Azul (BGR)
ORANGE = (0, 165, 255)      # Laranja (BGR)

# ---------------- estilo global ----------------
FONT_FACE         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SMALL  = 0.5     # Aumentado para melhor legibilidade
FONT_SCALE_LABEL  = 0.6     # Aumentado
THICK_TEXT        = 2       # Mais espesso para melhor visibilidade
THICK_BOX         = 2       # Boxes mais grossos
ALPHA_PANEL       = 0.75    # Painel mais opaco para melhor legibilidade
ALPHA_BANNER      = 0.65    # Banner mais opaco

# ---------------- helpers ----------------
def _draw_transparent_rect(img, x1, y1, x2, y2, fill=(0,0,0), alpha=0.35, border_color=(255,255,255), border_th=1):
    x1, y1 = int(max(0, x1)), int(max(0, y1))
    x2, y2 = int(min(img.shape[1]-1, x2)), int(min(img.shape[0]-1, y2))
    if x2 <= x1 or y2 <= y1: return
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    if border_th > 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, border_th, cv2.LINE_AA)

def _remove_accents(text: str) -> str:
    """
    Remove acentos de uma string para renderização no OpenCV.
    O OpenCV não suporta caracteres acentuados com fontes padrão.
    """
    # Mapeamento de caracteres acentuados para não acentuados
    accent_map = {
        'Á': 'A', 'À': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A',
        'á': 'a', 'à': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a',
        'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
        'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
        'Ó': 'O', 'Ò': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O',
        'ó': 'o', 'ò': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o',
        'Ú': 'U', 'Ù': 'U', 'Û': 'U', 'Ü': 'U',
        'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
        'Ç': 'C', 'ç': 'c',
    }
    result = ""
    for char in text:
        result += accent_map.get(char, char)
    return result

def _put_text(img, text, org, color=WHITE, scale=FONT_SCALE_SMALL, thick=THICK_TEXT, shadow=True):
    """Desenha texto com sombra para melhor legibilidade (remove acentos para compatibilidade com OpenCV)"""
    # Remove acentos antes de renderizar (OpenCV não suporta acentos)
    text_no_accents = _remove_accents(text)
    if shadow:
        # Sombra preta atrás do texto
        cv2.putText(img, text_no_accents, (org[0] + 1, org[1] + 1), FONT_FACE, scale, BLACK, thick + 1, cv2.LINE_AA)
    cv2.putText(img, text_no_accents, org, FONT_FACE, scale, color, thick, cv2.LINE_AA)

# Mapeamento de termos em inglês para português
_TRANSLATIONS = {
    "helmet": "Capacete",
    "gloves": "Luvas",
    "vest": "Colete",
    "apron": "Avental",
    "ear_protection": "Prot. Auricular",
    "helmet_color": "Capacete",
    "helmet_red": "Capacete",
    "helmet_blue": "Capacete",
    "helmet_yellow": "Capacete",
    "helmet_white": "Capacete",
    "helmet_brown": "Capacete",
}

# Mapeamento de cores em inglês para português
_COLOR_TRANSLATIONS = {
    "red": "vermelho",
    "blue": "azul",
    "yellow": "amarelo",
    "white": "branco",
    "gray": "cinza",
    "brown": "marrom",
    "red helmet": "vermelho",
    "blue helmet": "azul",
    "yellow helmet": "amarelo",
    "white helmet": "branco",
    "gray helmet": "cinza",
    "brown helmet": "marrom",
}

def _translate_key(key: str) -> str:
    """Traduz a chave do EPI para português"""
    return _TRANSLATIONS.get(key, key)

def _translate_color_value(value: str) -> str:
    """Traduz o valor da cor para português"""
    if value == "-":
        return "-"
    # Remove "helmet" se presente e traduz a cor
    for eng_color, pt_color in _COLOR_TRANSLATIONS.items():
        if eng_color in value.lower():
            return pt_color
    return value

def _status_to_text_color(val: Union[bool, str]):
    if isinstance(val, bool):
        return ("OK" if val else "NÃO"), (GREEN if val else RED)
    if isinstance(val, str) and val.upper().startswith("PEND"):
        return ("AGUARDANDO", YELLOW)
    return (str(val), WHITE)

# ---------------- Desenho ROI (overlay) ----------------
def draw_rois(img, polygons: List[dict], lines: List[dict], alpha: float = 0.25, debug: bool = False):
    """
    Desenha polígonos e linhas do ROI com preenchimento semi-transparente.
    polygons: [{"name": str, "pts": [(x,y),...]}]
    lines:    [{"name": str, "p1": (x,y), "p2": (x,y), "dir": int}]
    """
    if polygons:
        if debug:
            print(f"[DEBUG draw_rois] Recebidos {len(polygons)} polígono(s) para desenhar")
        overlay = img.copy()
        h, w = img.shape[:2]
        
        for poly in polygons:
            if not poly or "pts" not in poly:
                if debug:
                    print(f"[DEBUG draw_rois] Polígono inválido (sem 'pts'): {poly}")
                continue
                
            pts_list = poly.get("pts", [])
            if len(pts_list) < 3:
                if debug:
                    print(f"[DEBUG draw_rois] Polígono '{poly.get('name')}' tem menos de 3 pontos: {len(pts_list)}")
                continue
            
            # Converte e valida pontos
            # Primeiro, verifica se precisamos escalar os pontos
            # Se o ROI foi definido para uma resolução maior que a imagem atual
            pts = []
            for pt in pts_list:
                x, y = int(pt[0]), int(pt[1])
                # Clipping para garantir que está dentro da imagem
                x_clipped = max(0, min(x, w-1))
                y_clipped = max(0, min(y, h-1))
                pts.append([x_clipped, y_clipped])
            
            if len(pts) < 3:
                continue
            
            # Verifica se após o clipping ainda temos um polígono válido
            # (não todos os pontos no mesmo lugar)
            unique_pts = set((p[0], p[1]) for p in pts)
            if len(unique_pts) < 3:
                # Se todos os pontos ficaram no mesmo lugar após clipping, 
                # o ROI pode estar completamente fora da imagem
                if debug:
                    print(f"[DEBUG draw_rois] Polígono '{poly.get('name')}' inválido após clipping (todos pontos iguais)")
                continue
            
            if debug:
                min_x = min(p[0] for p in pts)
                max_x = max(p[0] for p in pts)
                min_y = min(p[1] for p in pts)
                max_y = max(p[1] for p in pts)
                print(f"[DEBUG draw_rois] Desenhando polígono '{poly.get('name')}': {len(pts)} pontos, bbox=({min_x},{min_y})-({max_x},{max_y}), img_size=({w},{h})")
                
            pts = np.array(pts, dtype=np.int32)
            name = (poly.get("name") or "").lower()
            
            # Cores mais vibrantes e visíveis
            if "insegur" in name or "danger" in name or "unsafe" in name:
                fill = (0, 0, 200);  border = (0, 0, 255)  # Vermelho mais vibrante
            elif "segur" in name or "safe" in name:
                fill = (0, 150, 0);  border = (0, 255, 0)  # Verde mais vibrante
            elif "epi" in name or "ppe" in name or "roi_epi" in name:
                fill = (0, 150, 255); border = (0, 200, 255)  # Laranja/Azul para EPI
            else:
                fill = (100, 100, 100); border = (200, 200, 200)  # Cinza mais claro
            
            # Desenha polígono preenchido
            cv2.fillPoly(overlay, [pts], fill)
            # Desenha borda do polígono (mais grossa e visível)
            cv2.polylines(overlay, [pts], isClosed=True, color=border, thickness=2, lineType=cv2.LINE_AA)
            
            # Label do polígono com fundo destacado
            x0, y0 = pts[0][0], pts[0][1]
            label_text = poly.get("name", "")
            if label_text:
                # Remove acentos para cálculo de tamanho e renderização (OpenCV não suporta acentos)
                label_text_clean = _remove_accents(label_text)
                (tw, th), baseline = cv2.getTextSize(label_text_clean, FONT_FACE, 0.5, 2)
                label_x = max(4, min(x0 + 4, w - tw - 4))
                label_y = max(th + 4, min(y0 - 4, h - 4))
                # Fundo do label
                cv2.rectangle(overlay, (label_x - 2, label_y - th - 2), 
                            (label_x + tw + 2, label_y + 2), BLACK, -1, cv2.LINE_AA)
                cv2.rectangle(overlay, (label_x - 2, label_y - th - 2), 
                            (label_x + tw + 2, label_y + 2), border, 1, cv2.LINE_AA)
                # Texto do label (já remove acentos automaticamente via _put_text)
                _put_text(overlay, label_text, (label_x, label_y), color=WHITE, scale=0.5, thick=2, shadow=False)
        
        # Aplica overlay com transparência
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    if lines:
        for ln in lines:
            p1 = tuple(map(int, ln["p1"])); p2 = tuple(map(int, ln["p2"]))
            # Linha mais grossa e visível
            cv2.line(img, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)
            # Marcador mais visível
            mx, my = int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)
            cv2.circle(img, (mx, my), 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (mx, my), 5, WHITE, 1, cv2.LINE_AA)
            # Label da linha
            label_text = ln.get("name", "")
            if label_text:
                # Remove acentos para cálculo de tamanho (OpenCV não suporta acentos)
                label_text_clean = _remove_accents(label_text)
                (tw, th), _ = cv2.getTextSize(label_text_clean, FONT_FACE, 0.5, 2)
                label_x = p1[0] + 4
                label_y = max(th + 4, p1[1] - 4)
                cv2.rectangle(img, (label_x - 2, label_y - th - 2), 
                            (label_x + tw + 2, label_y + 2), BLACK, -1, cv2.LINE_AA)
                cv2.rectangle(img, (label_x - 2, label_y - th - 2), 
                            (label_x + tw + 2, label_y + 2), (0, 255, 255), 1, cv2.LINE_AA)
                _put_text(img, label_text, (label_x, label_y), color=WHITE, scale=0.5, thick=2, shadow=False)

# ---------------- API ----------------
def draw_person_box(img, box, color=YELLOW, label="person"):
    """Desenha bounding box elegante com cantos destacados e label melhorado"""
    x1, y1, x2, y2 = map(int, box)
    
    # Desenha box principal com linha mais grossa
    cv2.rectangle(img, (x1, y1), (x2, y2), color, THICK_BOX, cv2.LINE_AA)
    
    # Desenha cantos destacados (estilo moderno)
    corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    corner_thick = max(2, THICK_BOX + 1)
    
    # Canto superior esquerdo
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, corner_thick, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, corner_thick, cv2.LINE_AA)
    
    # Canto superior direito
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, corner_thick, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, corner_thick, cv2.LINE_AA)
    
    # Canto inferior esquerdo
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, corner_thick, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, corner_thick, cv2.LINE_AA)
    
    # Canto inferior direito
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, corner_thick, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, corner_thick, cv2.LINE_AA)
    
    # Label melhorado
    text = label
    # Remove acentos para cálculo de tamanho (OpenCV não suporta acentos)
    text_clean = _remove_accents(text)
    (tw, th), baseline = cv2.getTextSize(text_clean, FONT_FACE, FONT_SCALE_LABEL, THICK_TEXT)
    pad_x, pad_y = 8, 6
    bx1 = x1
    by2 = max(y1 - 4, th + pad_y * 2)
    by1 = max(by2 - (th + pad_y * 2), 0)
    bx2 = min(bx1 + tw + pad_x * 2, img.shape[1] - 1)
    
    # Fundo do label com borda destacada
    _draw_transparent_rect(img, bx1, by1, bx2, by2, fill=color, alpha=0.85, border_color=color, border_th=2)
    # Borda interna branca para destaque
    cv2.rectangle(img, (bx1 + 1, by1 + 1), (bx2 - 1, by2 - 1), WHITE, 1, cv2.LINE_AA)
    
    # Texto com sombra
    text_x = bx1 + pad_x
    text_y = by2 - pad_y
    _put_text(img, text, (text_x, text_y), color=WHITE, scale=FONT_SCALE_LABEL, thick=THICK_TEXT, shadow=True)

def draw_ppe_panel(
    img,
    top_left: Tuple[int,int],
    results: Dict[str, Union[bool, str]],
    person_id: Optional[Union[int,str]] = None,
    roi_name: Optional[str] = None,
    font_scale: float = None,
    alert_status: Optional[str] = None
):
    """
    Painel elegante com fundo semi-transparente, bordas destacadas e melhor layout.
    Header: ID <id> e, se houver, "ROI: <nome>" logo abaixo.
    """
    if font_scale is None:
        font_scale = FONT_SCALE_SMALL

    x, y = top_left
    pad_x, pad_y = 10, 8
    line_gap = 4
    line_h = int(cv2.getTextSize("A", FONT_FACE, font_scale, THICK_TEXT)[0][1] + line_gap)

    items = list(results.items())

    # medir larguras (considerando traduções)
    maxw = 0
    for k, v in items:
        key_pt = _translate_key(k)
        if k == "helmet_color" and isinstance(v, str) and v != "-":
            color_pt = _translate_color_value(v)
            if color_pt != "-":
                text_str = f"{key_pt}: {color_pt}"
            else:
                s, _ = _status_to_text_color(v)
                text_str = f"{key_pt}: {s}"
        else:
            s, _ = _status_to_text_color(v)
            text_str = f"{key_pt}: {s}"
        # Remove acentos para cálculo de tamanho (OpenCV não suporta acentos)
        text_str_clean = _remove_accents(text_str)
        w, _ = cv2.getTextSize(text_str_clean, FONT_FACE, font_scale, THICK_TEXT)[0]
        maxw = max(maxw, w)

    # header (ID) e subheader (ROI)
    header_txt = f"ID {person_id}" if person_id is not None else None
    sub_txt = f"ROI: {roi_name}" if roi_name else None
    alert_txt = alert_status  # Status do alerta (se fornecido)

    header_h = 0
    if header_txt:
        # Remove acentos para cálculo de tamanho (OpenCV não suporta acentos)
        header_txt_clean = _remove_accents(header_txt)
        hw, hh = cv2.getTextSize(header_txt_clean, FONT_FACE, font_scale + 0.1, THICK_TEXT)[0]
        header_h += hh + line_gap + 4
        maxw = max(maxw, hw)
    if sub_txt:
        # Remove acentos para cálculo de tamanho (OpenCV não suporta acentos)
        sub_txt_clean = _remove_accents(sub_txt)
        sw, sh = cv2.getTextSize(sub_txt_clean, FONT_FACE, font_scale*0.95, THICK_TEXT)[0]
        header_h += sh + line_gap + 2
        maxw = max(maxw, sw)
    if alert_txt:
        # Remove acentos para cálculo de tamanho (OpenCV não suporta acentos)
        alert_txt_clean = _remove_accents(alert_txt)
        aw, ah = cv2.getTextSize(alert_txt_clean, FONT_FACE, font_scale*0.9, THICK_TEXT)[0]
        header_h += ah + line_gap + 2
        maxw = max(maxw, aw)

    panel_w = pad_x*2 + maxw + 4
    panel_h = pad_y*2 + line_h*len(items) + header_h + 4

    # clamp na tela
    x = int(min(max(0, x), img.shape[1]-panel_w-1))
    y = int(min(max(0, y), img.shape[0]-panel_h-1))

    # Fundo principal do painel com borda destacada
    _draw_transparent_rect(img, x, y, x+panel_w, y+panel_h, fill=BLACK, alpha=ALPHA_PANEL, border_color=WHITE, border_th=2)
    # Borda externa colorida (azul)
    cv2.rectangle(img, (x-1, y-1), (x+panel_w+1, y+panel_h+1), BLUE, 1, cv2.LINE_AA)

    # header e subheader com fundo destacado
    yy = y + pad_y
    if header_txt:
        hb_y2 = yy + int(header_h*0.6)
        # Fundo do header com gradiente escuro
        _draw_transparent_rect(img, x+2, yy, x+panel_w-2, hb_y2, fill=(20, 20, 50), alpha=0.9, border_color=BLUE, border_th=1)
        _put_text(img, header_txt, (x + pad_x, yy + int((hb_y2-yy)*0.7)), color=LIGHT, scale=font_scale + 0.1, thick=THICK_TEXT, shadow=True)
        yy = hb_y2 + 3
    if sub_txt:
        _put_text(img, sub_txt, (x + pad_x, yy + int(line_h*0.9)), color=(200, 200, 255), scale=font_scale*0.95, thick=THICK_TEXT, shadow=True)
        yy += line_h + 2
    if alert_txt:
        # Determina cor baseado no status do alerta
        alert_txt_upper = alert_txt.upper()
        if "ALERTA GERADO" in alert_txt_upper or "ENVIADO" in alert_txt_upper:
            alert_color = RED
        elif "SUPRIMIDO" in alert_txt_upper:
            alert_color = YELLOW
        elif "AGUARDANDO" in alert_txt_upper or "PENDENTE" in alert_txt_upper:
            alert_color = ORANGE
        elif "VIOLAÇÃO ATIVA" in alert_txt_upper:
            alert_color = RED
        else:
            alert_color = (200, 200, 200)  # Cinza claro
        
        _put_text(img, alert_txt, (x + pad_x, yy + int(line_h*0.9)), color=alert_color, scale=font_scale*0.9, thick=THICK_TEXT, shadow=True)
        yy += line_h + 2
        # Linha separadora
        cv2.line(img, (x + pad_x, yy), (x + panel_w - pad_x, yy), (100, 100, 100), 1, cv2.LINE_AA)
        yy += 3
    elif sub_txt:
        # Linha separadora apenas se não houver alert_txt
        cv2.line(img, (x + pad_x, yy), (x + panel_w - pad_x, yy), (100, 100, 100), 1, cv2.LINE_AA)
        yy += 3

    # linhas de status com ícones visuais (traduzidas para português)
    for k, v in items:
        # Traduz a chave
        key_pt = _translate_key(k)
        
        # Trata helmet_color de forma especial
        if k == "helmet_color" and isinstance(v, str) and v != "-":
            # Formato: "Capacete: amarelo" ao invés de "helmet_color: yellow helmet"
            color_pt = _translate_color_value(v)
            if color_pt != "-":
                text_str = f"{key_pt}: {color_pt}"
                # Para helmet_color com cor detectada, usa verde
                color = GREEN
            else:
                s, color = _status_to_text_color(v)
                text_str = f"{key_pt}: {s}"
        else:
            s, color = _status_to_text_color(v)
            text_str = f"{key_pt}: {s}"
        
        # Desenha círculo indicador antes do texto
        circle_x = x + pad_x - 8
        circle_y = yy + line_h - line_gap - 3
        circle_radius = 4
        cv2.circle(img, (circle_x, circle_y), circle_radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (circle_x, circle_y), circle_radius + 1, WHITE, 1, cv2.LINE_AA)
        
        # Texto com melhor formatação
        _put_text(img, text_str, (x + pad_x, yy + line_h - line_gap), color=color, scale=font_scale, thick=THICK_TEXT, shadow=True)
        yy += line_h

def put_banner(img, text:str, color=WHITE, bg=(20, 20, 20)):
    """Banner melhorado com borda e sombra"""
    # Remove acentos para cálculo de tamanho (OpenCV não suporta acentos)
    text_clean = _remove_accents(text)
    (tw, th), baseline = cv2.getTextSize(text_clean, FONT_FACE, FONT_SCALE_SMALL, THICK_TEXT)
    pad_y = 8
    pad_x = 12
    h = th + pad_y*2
    w = img.shape[1]
    
    # Fundo do banner
    _draw_transparent_rect(img, 0, 0, w-1, h, fill=bg, alpha=ALPHA_BANNER, border_color=WHITE, border_th=2)
    # Linha decorativa inferior
    cv2.line(img, (0, h-1), (w-1, h-1), BLUE, 2, cv2.LINE_AA)
    
    # Texto com sombra
    _put_text(img, text, (pad_x, h - pad_y), color=color, scale=FONT_SCALE_SMALL, thick=THICK_TEXT, shadow=True)

def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w-1))
    y1 = max(0, min(int(y1), h-1))
    x2 = max(0, min(int(x2), w-1))
    y2 = max(0, min(int(y2), h-1))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return [x1, y1, x2, y2]

def draw_metrics_overlay(img, metrics: Dict[str, float], position: Tuple[int, int] = (10, 30)):
    """
    Desenha overlay de métricas de performance no frame.
    
    Args:
        img: Frame BGR
        metrics: Dict com métricas em ms: {"rtdetr_ms": 50.0, "ppe_ms": 200.0, "track_ms": 10.0, "draw_ms": 15.0, "total_ms": 275.0}
        position: Posição (x, y) do canto superior esquerdo do overlay
    """
    if not metrics:
        return
    
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 18
    bg_alpha = 0.7
    
    # Lista de métricas para exibir (em ordem)
    metric_labels = [
        ("RT-DETR", "rtdetr_ms", GREEN),
        ("PPE Det", "ppe_ms", YELLOW),
        ("Track", "track_ms", BLUE),
        ("Draw", "draw_ms", ORANGE),
        ("Total", "total_ms", WHITE),
    ]
    
    # Calcula altura total do overlay
    num_lines = len([m for m in metric_labels if m[1] in metrics])
    overlay_height = num_lines * line_height + 10
    overlay_width = 150
    
    # Desenha fundo semi-transparente
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y - 5), (x + overlay_width, y + overlay_height), BLACK, -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    
    # Desenha borda
    cv2.rectangle(img, (x, y - 5), (x + overlay_width, y + overlay_height), WHITE, 1)
    
    # Desenha cada métrica
    current_y = y + 15
    for label, key, color in metric_labels:
        if key in metrics:
            value = metrics[key]
            text = f"{label}: {value:.1f}ms"
            cv2.putText(img, text, (x + 5, current_y), font, font_scale, color, thickness, cv2.LINE_AA)
            current_y += line_height

def draw_alert_grid(img, grid_size: int = 8, suppressed_cells: List[Tuple[int, int]] = None, 
                    violation_cells: List[Tuple[int, int]] = None, alpha: float = 0.5):
    """
    Desenha grid 8x8 estilo tabuleiro de xadrez com perspectiva 3D.
    
    Args:
        img: Imagem OpenCV (BGR)
        grid_size: Tamanho do grid (8x8)
        suppressed_cells: Lista de (grid_x, grid_y) com alertas suprimidos (vermelho)
        violation_cells: Lista de (grid_x, grid_y) com violações ativas (amarelo/laranja)
        alpha: Transparência do grid
    """
    if suppressed_cells is None:
        suppressed_cells = []
    if violation_cells is None:
        violation_cells = []
    
    H, W = img.shape[:2]
    cell_w = W / grid_size
    cell_h = H / grid_size
    
    # Cores
    COLOR_WHITE = (200, 200, 200)  # Branco claro (BGR)
    COLOR_BLACK = (50, 50, 50)     # Preto escuro (BGR)
    COLOR_SUPPRESSED = (0, 0, 255)  # Vermelho para alertas suprimidos
    COLOR_VIOLATION = (0, 165, 255)  # Laranja para violações ativas
    
    # Cria overlay separado para aplicar transparência por célula
    overlay = np.zeros_like(img)
    
    for gy in range(grid_size):
        for gx in range(grid_size):
            # Calcula posição da célula
            x1 = int(gx * cell_w)
            y1 = int(gy * cell_h)
            x2 = int((gx + 1) * cell_w)
            y2 = int((gy + 1) * cell_h)
            
            # Determina cor base
            is_white = (gx + gy) % 2 == 0
            base_color = COLOR_WHITE if is_white else COLOR_BLACK
            
            # Verifica se célula tem alerta suprimido (prioridade máxima)
            if (gx, gy) in suppressed_cells:
                cell_color = COLOR_SUPPRESSED
                # Células suprimidas são mais opacas
                use_alpha = 0.6
            elif (gx, gy) in violation_cells:
                cell_color = COLOR_VIOLATION
                # Células com violação são mais opacas
                use_alpha = 0.5
            else:
                cell_color = base_color
                use_alpha = alpha
            
            # Desenha célula no overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), cell_color, -1)
            
            # Efeito 3D: borda superior e esquerda mais clara, inferior e direita mais escura
            if cell_color == base_color:  # Só aplica efeito 3D nas células normais
                # Borda superior (clara)
                cv2.line(overlay, (x1, y1), (x2, y1), (min(255, cell_color[0] + 40), 
                                                       min(255, cell_color[1] + 40), 
                                                       min(255, cell_color[2] + 40)), 1)
                # Borda esquerda (clara)
                cv2.line(overlay, (x1, y1), (x1, y2), (min(255, cell_color[0] + 40), 
                                                       min(255, cell_color[1] + 40), 
                                                       min(255, cell_color[2] + 40)), 1)
                # Borda inferior (escura)
                cv2.line(overlay, (x1, y2), (x2, y2), (max(0, cell_color[0] - 40), 
                                                       max(0, cell_color[1] - 40), 
                                                       max(0, cell_color[2] - 40)), 1)
                # Borda direita (escura)
                cv2.line(overlay, (x2, y1), (x2, y2), (max(0, cell_color[0] - 40), 
                                                       max(0, cell_color[1] - 40), 
                                                       max(0, cell_color[2] - 40)), 1)
            else:
                # Para células com alerta, desenha borda destacada
                cv2.rectangle(overlay, (x1, y1), (x2, y2), cell_color, 2)
            
            # Aplica transparência por célula diretamente na imagem
            cell_roi = overlay[y1:y2, x1:x2]
            img_roi = img[y1:y2, x1:x2]
            cv2.addWeighted(cell_roi, use_alpha, img_roi, 1 - use_alpha, 0, img_roi)
