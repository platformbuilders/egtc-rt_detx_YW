"""
Camera Thread - Thread que processa uma câmera individual.
Baseado no loop principal do pipeline_RETDETRX_YW.py, mas adaptado para threading.
"""
import os
import time
import threading
import traceback
import cv2
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from pipeline_RETDETRX_YW import (
    Camera, ROI, PPETracker, AlertManager, AlertConfig,
    _eval_flags_from_frame, load_yaml, read_prompts
)
from utils import (
    draw_person_box, draw_ppe_panel, put_banner, draw_rois, draw_alert_grid,
    GREEN, RED, YELLOW, ORANGE, draw_metrics_overlay
)
from logger import get_logger


class CameraThread(threading.Thread):
    """
    Thread que processa uma câmera individual.
    Carrega modelos compartilhados e processa frames em loop.
    """
    
    def __init__(self,
                 camera_id: str,
                 camera_uri: str,
                 camera_config: Dict[str, Any],
                 global_config: Dict[str, Any],
                 prompts: Dict[str, Any],
                 person_detector: Any,
                 ppe_detector: Any,
                 shared_frame_buffer: Dict[str, bytes],
                 frame_buffer_lock: threading.Lock,
                 running_flag: threading.Event):
        """
        Inicializa a thread de câmera.
        
        Args:
            camera_id: ID da câmera (ex: "CAM063")
            camera_uri: URI da câmera (RTSP ou arquivo)
            camera_config: Configuração específica da câmera (roi_path, roi_polys, roi_ppe_config)
            global_config: Configuração global do YAML
            prompts: Prompts de EPIs (positive/negative)
            person_detector: Detector de pessoas (RT-DETR ou YOLO) - compartilhado
            ppe_detector: Detector de EPIs (YOLO-World ou OWL-V2) - compartilhado
            shared_frame_buffer: Dicionário compartilhado {camera_id: frame_bytes} para streaming
            frame_buffer_lock: Lock para acesso thread-safe ao buffer
            running_flag: Event para controlar execução (set() = parar)
        """
        super().__init__(name=f"CameraThread-{camera_id}", daemon=False)
        self.camera_id = camera_id
        self.camera_uri = camera_uri
        self.camera_config = camera_config
        self.global_config = global_config
        self.prompts = prompts
        self.person_detector = person_detector
        self.ppe_detector = ppe_detector
        self.shared_frame_buffer = shared_frame_buffer
        self.frame_buffer_lock = frame_buffer_lock
        self.running_flag = running_flag
        
        self.logger = get_logger()
        
        # Configurações locais
        self.debug = bool(global_config.get("debug", False))
        self.target_fps = float(global_config.get("target_fps", 1.0))
        self.out_dir = global_config.get("out_dir", "./out")
        self.save_video = bool(global_config.get("save_video", True))
        self.video_fps = int(global_config.get("video_fps", 2))
        self.show_video = bool(global_config.get("show_video", False))
        self.show_rtdetr_boxes = bool(global_config.get("show_rtdetr_boxes", False))
        self.draw_roi = bool(global_config.get("draw_roi", False))
        self.metrics_overlay = bool(global_config.get("metrics_overlay", True))
        self.metrics_csv = bool(global_config.get("metrics_csv", True))
        self.metrics_print_every = int(global_config.get("metrics_print_every", 30))
        
        # Threshold PPE
        ppe_detector_type = global_config.get("ppe_detector", "yolo-world").lower()
        if ppe_detector_type == "yolo-world":
            self.ppe_score_thr = float(global_config.get("yw_score_thr", 0.15))
        else:
            self.ppe_score_thr = float(global_config.get("ovd_score_thr", 0.26))
        
        # Tracking
        debounce_sec = float(global_config.get("debounce_seconds", 8.0))
        track_thresh = float(global_config.get("track_thresh", 0.25))
        match_thresh = float(global_config.get("match_thresh", 0.3))
        track_buffer = int(global_config.get("track_buffer", 60))
        iou_thresh = float(global_config.get("track_iou_thresh", 0.3))
        max_age = int(global_config.get("track_max_age", 30))
        
        # Zonas verticais
        self.head_ratio = float(global_config.get("head_ratio", 0.45))
        self.chest_min = float(global_config.get("chest_min_ratio", 0.35))
        self.chest_max = float(global_config.get("chest_max_ratio", 0.75))
        
        # ROI - configuração específica da câmera
        roi_path = camera_config.get("roi_path")
        if roi_path:
            # A classe ROI resolve caminhos relativos ao diretório do pipeline_RETDETRX_YW.py
            # que é o mesmo diretório raiz do projeto. Se o caminho for relativo, passamos
            # diretamente. Se for absoluto, passamos como está.
            # Não precisamos resolver aqui, a classe ROI já faz isso.
            pass
        roi_polys = camera_config.get("roi_polys", [])
        self.roi = ROI(roi_path, use_polygons=roi_polys, debug=self.debug) if roi_path else ROI(None, debug=self.debug)
        
        # Configuração de EPIs por ROI (específica da câmera ou global)
        self.roi_ppe_config = camera_config.get("roi_ppe_config", {})
        if not self.roi_ppe_config:
            self.roi_ppe_config = global_config.get("global_roi_ppe_config", {})
        
        # EPIs obrigatórios padrão
        self.req_ppe_default = global_config.get("required_ppe", ["helmet", "gloves", "ear_protection", "vest"])
        
        # Sistema de alertas
        self.enable_alerts = bool(global_config.get("enable_alerts", False))
        self.show_alert_grid = bool(global_config.get("show_alert_grid", False))
        self.alert_config_path = global_config.get("alert_config_path", "db_config.env")
        self.alert_manager: Optional[AlertManager] = None
        
        # Inicializa componentes
        self.camera = Camera(self.camera_id, self.camera_uri, self.target_fps, debug=self.debug)
        self.tracker = PPETracker(
            fps_hint=self.target_fps,
            debounce_seconds=debounce_sec,
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            iou_thresh=iou_thresh,
            max_age=max_age
        )
        
        # Contadores e métricas
        self.frame_counter = 0
        self.last_metrics = {}
        
        # Estado
        self.error_count = 0
        self.max_errors = 10
        
        self.logger.info(self.camera_id, f"Thread inicializada - ROI: {roi_path or 'N/A'}, EPIs por ROI: {list(self.roi_ppe_config.keys())}")
    
    def run(self):
        """Loop principal de processamento da câmera."""
        self.logger.info(self.camera_id, "Thread iniciada")
        
        try:
            while not self.running_flag.is_set():
                try:
                    if not self.camera.should_grab():
                        time.sleep(0.01)
                        continue
                    
                    t_total0 = time.perf_counter()
                    ok, frame = self.camera.grab()
                    if not ok or frame is None:
                        if self.debug:
                            self.logger.debug(self.camera_id, f"Grab falhou. ok={ok}, frame=None")
                        time.sleep(0.1)
                        continue
                    
                    H, W = frame.shape[:2]
                    self.frame_counter += 1
                    
                    # Validação de resolução ROI (primeiro frame)
                    if self.frame_counter == 1 and self.roi.active:
                        self.roi.validate_resolution(frame.shape)
                    
                    # Inicializa AlertManager no primeiro frame
                    if (self.show_alert_grid or self.enable_alerts) and self.alert_manager is None:
                        alert_config = AlertConfig.from_env_file(self.alert_config_path)
                        # Ajusta configurações do YAML
                        alert_config.alert_debounce_seconds = float(self.global_config.get("alert_debounce_seconds", alert_config.alert_debounce_seconds))
                        alert_config.alert_min_consecutive_frames = int(self.global_config.get("alert_min_consecutive_frames", alert_config.alert_min_consecutive_frames))
                        alert_config.suppression_reset_seconds = float(self.global_config.get("alert_suppression_reset_seconds", alert_config.suppression_reset_seconds))
                        alert_config.alert_hash_ttl_seconds = float(self.global_config.get("alert_hash_ttl_seconds", alert_config.alert_hash_ttl_seconds))
                        alert_config.grid_size = int(self.global_config.get("alert_grid_size", alert_config.grid_size))
                        alert_config.timezone_offset_hours = float(self.global_config.get("timezone_offset_hours", alert_config.timezone_offset_hours))
                        alert_config.save_alert_images = bool(self.global_config.get("save_alert_images", alert_config.save_alert_images))
                        alert_config.save_crop_only = bool(self.global_config.get("save_crop_only", alert_config.save_crop_only))
                        if "crops_dir" in self.global_config:
                            alert_config.crops_dir = str(self.global_config["crops_dir"])
                        self.alert_manager = AlertManager(alert_config, self.camera_id, W, H, send_alerts=self.enable_alerts)
                        self.logger.info(self.camera_id, f"AlertManager inicializado - resolução {W}x{H}, envio={'ON' if self.enable_alerts else 'OFF'}")
                    
                    # ==== RT-DETR-X (detecção de pessoas) ====
                    t1 = time.perf_counter()
                    boxes = self.person_detector.detect(frame, debug=self.debug)
                    t2 = time.perf_counter()
                    rtdetr_ms = (t2 - t1) * 1000.0
                    
                    if self.debug and self.frame_counter % 30 == 0:
                        self.logger.debug(self.camera_id, f"RT-DETR: {len(boxes)} pessoa(s) detectada(s)")
                    
                    # ==== Filtro ROI ====
                    if self.roi.active:
                        boxes_before_roi = len(boxes)
                        boxes = [box for box in boxes if self.roi.contains_box([int(box[0]), int(box[1]), int(box[2]), int(box[3])], min_overlap_ratio=0.3)]
                        if self.debug and boxes_before_roi != len(boxes) and self.frame_counter % 30 == 0:
                            self.logger.debug(self.camera_id, f"Filtro ROI: {boxes_before_roi} -> {len(boxes)} pessoa(s)")
                    
                    raw_boxes_for_debug = list(boxes) if boxes else []
                    
                    # ==== TRACKER ====
                    t3 = time.perf_counter()
                    dets = np.array([[x1, y1, x2, y2, conf] for (x1, y1, x2, y2, conf) in boxes], dtype=float) if boxes else np.zeros((0, 5), dtype=float)
                    tracks = self.tracker.update(dets, frame_size=frame.shape[:2])
                    t4 = time.perf_counter()
                    track_ms = (t4 - t3) * 1000.0
                    
                    persons = len(boxes)
                    n_tracks = len(tracks)
                    
                    # ==== Detecção de EPIs ====
                    t7 = time.perf_counter()
                    person_boxes = [[int(x1), int(y1), int(x2), int(y2)] for (x1, y1, x2, y2, _) in boxes] if boxes else []
                    pos = self.prompts.get("positive", {})
                    neg = self.prompts.get("negative", {})
                    
                    _, raw_frame = self.ppe_detector.infer(
                        frame,
                        pos,
                        score_thr=self.ppe_score_thr,
                        person_boxes=person_boxes if self.ppe_detector.use_crop else None,
                        negative=neg
                    )
                    t8 = time.perf_counter()
                    ppe_ms = (t8 - t7) * 1000.0
                    
                    # ==== ROI overlay ====
                    if self.draw_roi and self.roi.active:
                        draw_rois(frame, self.roi.polygons, self.roi.lines, debug=self.debug)
                    
                    # ==== Atribuição por track + debounce + ROI gating ====
                    t9 = time.perf_counter()
                    people_normal = []
                    people_violators = []
                    
                    for tr in tracks:
                        tid, pbox = tr["id"], list(map(int, tr["box"]))
                        cx, cy = (0.5 * (pbox[0] + pbox[2]), 0.5 * (pbox[1] + pbox[3]))
                        inside = self.roi.contains_box(pbox, min_overlap_ratio=0.3) if self.roi.active else True
                        roi_names = self.roi.which((cx, cy)) if self.roi.active else []
                        roi_name_txt = ", ".join(roi_names) if roi_names else None
                        
                        if inside:
                            flags, helmet_color = _eval_flags_from_frame(
                                raw_frame, pos, pbox, self.head_ratio, self.chest_min, self.chest_max,
                                frame_bgr=frame, debug=self.debug
                            )
                            debounced = self.tracker.update_ppe(tid, flags)
                            
                            # Determina EPIs obrigatórios baseado no ROI
                            req_ppe = []
                            if self.roi.active:
                                if roi_names and self.roi_ppe_config:
                                    for roi_name in roi_names:
                                        if roi_name in self.roi_ppe_config:
                                            req_ppe = self.roi_ppe_config[roi_name]
                                            break
                            else:
                                req_ppe = self.req_ppe_default
                            
                            # Verifica violações
                            is_violation = False
                            violation_items = []
                            
                            if req_ppe:
                                for req_item in req_ppe:
                                    is_item_violation = False
                                    if req_item == "helmet":
                                        if debounced.get("helmet") is False:
                                            is_violation = True
                                            is_item_violation = True
                                    elif req_item.startswith("helmet_"):
                                        if debounced.get(req_item) is False:
                                            is_violation = True
                                            is_item_violation = True
                                    else:
                                        if debounced.get(req_item) is False:
                                            is_violation = True
                                            is_item_violation = True
                                    
                                    if is_item_violation:
                                        violation_items.append(req_item)
                            
                            # Cria ppe_flags_display apenas com EPIs monitorados
                            ppe_flags_display = {}
                            if req_ppe:
                                for item in req_ppe:
                                    if item == "helmet":
                                        ppe_flags_display["helmet"] = debounced.get("helmet", "PENDING")
                                        if debounced.get("helmet") is True:
                                            ppe_flags_display["helmet_color"] = (helmet_color if helmet_color != "-" else "-")
                                        else:
                                            ppe_flags_display["helmet_color"] = "-"
                                    elif item.startswith("helmet_"):
                                        ppe_flags_display["helmet"] = debounced.get("helmet", "PENDING")
                                        ppe_flags_display[item] = debounced.get(item, "PENDING")
                                        if debounced.get("helmet") is True:
                                            ppe_flags_display["helmet_color"] = (helmet_color if helmet_color != "-" else "-")
                                        else:
                                            ppe_flags_display["helmet_color"] = "-"
                                    else:
                                        ppe_flags_display[item] = debounced.get(item, "PENDING")
                            
                            if is_violation:
                                people_violators.append((tid, pbox, ppe_flags_display, roi_name_txt))
                            else:
                                people_normal.append((tid, pbox, ppe_flags_display, roi_name_txt))
                        else:
                            people_normal.append((tid, pbox, None, roi_name_txt))
                    
                    t10 = time.perf_counter()
                    agg_ms = (t10 - t9) * 1000.0
                    
                    # ==== Sistema de Alertas ====
                    alerts_generated = []
                    if (self.show_alert_grid or self.enable_alerts) and self.alert_manager:
                        violations_for_alert = []
                        for tid, pbox, ppe_flags, roi_name_txt in people_violators:
                            missing_ppe = []
                            if ppe_flags:
                                for epi_key, epi_status in ppe_flags.items():
                                    if epi_key != "helmet_color" and epi_status is False:
                                        missing_ppe.append(epi_key)
                            if missing_ppe:
                                violations_for_alert.append((tid, pbox, missing_ppe))
                        
                        if violations_for_alert:
                            self.alert_manager.update_violations(violations_for_alert)
                        
                        alerts_generated = self.alert_manager.check_and_generate_alerts(
                            frame_bgr=frame if self.enable_alerts else None
                        )
                    
                    # ==== Desenho ====
                    t_draw_start = time.perf_counter()
                    frame_drawn = frame.copy()
                    
                    # Desenha boxes RT-DETR (debug)
                    if self.show_rtdetr_boxes and raw_boxes_for_debug:
                        for x1, y1, x2, y2, conf in raw_boxes_for_debug:
                            offset = 5
                            x1_i = max(0, int(x1) - offset)
                            y1_i = max(0, int(y1) - offset)
                            x2_i = min(W - 1, int(x2) + offset)
                            y2_i = min(H - 1, int(y2) + offset)
                            cv2.rectangle(frame_drawn, (x1_i, y1_i), (x2_i, y2_i), ORANGE, 3, cv2.LINE_AA)
                            from utils import _remove_accents
                            label_text = _remove_accents(f"RT-DETR {conf:.2f}")
                            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame_drawn, (x1_i, y1_i - th - 8), (x1_i + tw + 8, y1_i), ORANGE, -1, cv2.LINE_AA)
                            cv2.putText(frame_drawn, label_text, (x1_i + 4, y1_i - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Desenha grid de alertas
                    if self.show_alert_grid and self.alert_manager:
                        suppressed_cells = self.alert_manager.get_suppressed_cells()
                        violation_cells = []
                        for track_id, state in self.alert_manager.violation_states.items():
                            if not state.get("alerted", False):
                                violation_cells.append((state["grid_x"], state["grid_y"]))
                        draw_alert_grid(frame_drawn, grid_size=self.alert_manager.config.grid_size,
                                       suppressed_cells=suppressed_cells,
                                       violation_cells=violation_cells)
                    
                    # Desenha pessoas normais
                    for tid, box, ppe_flags, roi_name_txt in people_normal:
                        draw_person_box(frame_drawn, box, color=GREEN, label=f"ID {tid}")
                        if ppe_flags:
                            px = min(int(box[0]) + 5, W - 260)
                            py = min(int(box[3]) + 5, H - 200)
                            alert_status = None
                            if (self.show_alert_grid or self.enable_alerts) and self.alert_manager:
                                cx = (box[0] + box[2]) / 2
                                cy = (box[1] + box[3]) / 2
                                alert_status = self.alert_manager.get_alert_status(cx, cy, track_id=tid)
                            draw_ppe_panel(frame_drawn, (px, py), ppe_flags, person_id=tid, roi_name=roi_name_txt, font_scale=0.55, alert_status=alert_status)
                    
                    # Desenha violadores
                    for tid, box, ppe_flags, roi_name_txt in people_violators:
                        box_color = RED
                        box_label = f"VIOLAÇÃO ID {tid}"
                        
                        if (self.show_alert_grid or self.enable_alerts) and self.alert_manager:
                            cx = (box[0] + box[2]) / 2
                            cy = (box[1] + box[3]) / 2
                            alert_status = self.alert_manager.get_alert_status(cx, cy, track_id=tid)
                            
                            if alert_status == "ALERTA GERADO":
                                box_color = RED
                                box_label = f"ALERTA ID {tid}"
                            else:
                                box_color = YELLOW
                                box_label = f"AVALIANDO ID {tid}"
                        
                        draw_person_box(frame_drawn, box, color=box_color, label=box_label)
                        
                        if ppe_flags:
                            px = min(int(box[0]) + 5, W - 260)
                            py = min(int(box[3]) + 5, H - 200)
                            alert_status = None
                            if (self.show_alert_grid or self.enable_alerts) and self.alert_manager:
                                cx = (box[0] + box[2]) / 2
                                cy = (box[1] + box[3]) / 2
                                alert_status = self.alert_manager.get_alert_status(cx, cy, track_id=tid)
                            draw_ppe_panel(frame_drawn, (px, py), ppe_flags, person_id=tid, roi_name=roi_name_txt, font_scale=0.55, alert_status=alert_status)
                    
                    # Banner
                    t_total_end = time.perf_counter()
                    total_ms = (t_total_end - t_total0) * 1000.0
                    
                    violation_status = None
                    violation_color = None
                    if people_violators:
                        if alerts_generated and len(alerts_generated) > 0:
                            violation_status = "ALERTA"
                            violation_color = RED
                        else:
                            violation_status = "VIOLACAO"
                            violation_color = YELLOW
                    elif alerts_generated and len(alerts_generated) > 0:
                        violation_status = "ALERTA"
                        violation_color = RED
                    
                    if self.metrics_overlay:
                        banner = (f"{self.camera_id} | pessoas {persons} | rastros {len(tracks)} | "
                                  f"RT-DETR {rtdetr_ms:.0f}ms | rastr {track_ms:.0f}ms | "
                                  f"PPE {ppe_ms:.0f}ms | agg {agg_ms:.0f}ms | "
                                  f"ROI {'ligado' if self.roi.active else 'desligado'} | total {total_ms:.0f}ms")
                    else:
                        banner = f"{self.camera_id}  |  pessoas: {persons}  |  rastros: {len(tracks)}"
                    
                    if violation_status:
                        banner = f"{banner}  |  {violation_status}"
                    
                    if violation_color:
                        put_banner(frame_drawn, banner, color=violation_color)
                    else:
                        put_banner(frame_drawn, banner)
                    
                    # Métricas overlay
                    if self.metrics_overlay:
                        all_metrics = {
                            "rtdetr_ms": rtdetr_ms,
                            "ppe_ms": ppe_ms,
                            "track_ms": track_ms,
                            "draw_ms": (time.perf_counter() - t_draw_start) * 1000.0,
                            "total_ms": total_ms
                        }
                        draw_metrics_overlay(frame_drawn, all_metrics, position=(10, 30))
                    
                    # ==== Exibição em tempo real ====
                    if self.show_video:
                        window_name = f"Pipeline - {self.camera_id}"
                        cv2.imshow(window_name, frame_drawn)
                        cv2.waitKey(1)
                    
                    # ==== Gravação ====
                    if self.save_video:
                        self.camera.ensure_writer(self.out_dir, self.video_fps, frame_drawn.shape)
                        self.camera.write(frame_drawn, force_jpg=self.debug)
                    elif self.debug:
                        os.makedirs(self.out_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(self.out_dir, f"{self.camera_id}_latest.jpg"), frame_drawn)
                    
                    # ==== CSV ====
                    if self.metrics_csv:
                        self.camera.log_metrics(self.out_dir, [
                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            self.camera_id, persons, len(tracks), len(person_boxes) if self.ppe_detector.use_crop else 1,
                            f"{rtdetr_ms:.2f}", f"{track_ms:.2f}", "0.00",
                            f"{ppe_ms:.2f}", f"{agg_ms:.2f}", "0.00",
                            "0.00", f"{total_ms:.2f}"
                        ])
                    
                    # ==== Atualiza buffer compartilhado para streaming ====
                    try:
                        frame_bytes = pickle.dumps(frame_drawn)
                        with self.frame_buffer_lock:
                            self.shared_frame_buffer[self.camera_id] = frame_bytes
                    except Exception as e:
                        if self.frame_counter % 30 == 0:
                            self.logger.warning(self.camera_id, f"Erro ao atualizar buffer de streaming: {e}")
                    
                    # ==== Print periódico ====
                    if self.metrics_print_every > 0 and (self.frame_counter % self.metrics_print_every == 0):
                        self.logger.info(self.camera_id, f"Frame {self.frame_counter} | RT-DETR {rtdetr_ms:.1f}ms | trk {track_ms:.1f}ms | PPE {ppe_ms:.1f}ms | total {total_ms:.1f}ms")
                    
                    # Reset error count em caso de sucesso
                    self.error_count = 0
                    
                except Exception as e:
                    self.error_count += 1
                    self.logger.error(self.camera_id, f"Erro no processamento: {e}")
                    if self.debug:
                        self.logger.exception(self.camera_id, f"Traceback: {traceback.format_exc()}")
                    
                    if self.error_count >= self.max_errors:
                        self.logger.critical(self.camera_id, f"Muitos erros consecutivos ({self.error_count}). Parando thread.")
                        break
                    
                    time.sleep(0.5)
        
        except KeyboardInterrupt:
            self.logger.info(self.camera_id, "Interrompido pelo usuário")
        except Exception as e:
            self.logger.critical(self.camera_id, f"Erro fatal na thread: {e}")
            self.logger.exception(self.camera_id, f"Traceback: {traceback.format_exc()}")
        finally:
            self._cleanup()
            self.logger.info(self.camera_id, "Thread finalizada")
    
    def _cleanup(self):
        """Limpa recursos da câmera."""
        try:
            if self.camera:
                self.camera.close()
        except Exception as e:
            self.logger.error(self.camera_id, f"Erro ao limpar recursos: {e}")

