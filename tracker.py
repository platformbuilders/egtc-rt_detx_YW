# tracker.py
import time
from typing import List, Tuple, Dict, Optional
import numpy as np

try:
    # Ultralytics ByteTrack (assinaturas variam por versão)
    from ultralytics.trackers.byte_tracker import BYTETracker  # type: ignore
    _HAS_ULT_BYTE = True
except Exception:
    _HAS_ULT_BYTE = False


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (boxAArea + boxBArea - inter + 1e-6)


class SimpleIoUTracker:
    """Fallback tracker quando BYTETracker não está disponível. IDs estáveis com associação por IoU."""
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 15):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}  # id -> dict(box, age)

    def update(self, dets: np.ndarray):
        # dets: [N,5] -> x1,y1,x2,y2,conf
        assigned = set()
        updates = {}
        for tid, tr in list(self.tracks.items()):
            best_iou, best_j = 0.0, -1
            for j, d in enumerate(dets):
                if j in assigned:
                    continue
                ov = iou(tr["box"], d[:4])
                if ov > best_iou:
                    best_iou, best_j = ov, j
            if best_iou >= self.iou_thresh and best_j >= 0:
                self.tracks[tid]["box"] = dets[best_j][:4].tolist()
                self.tracks[tid]["age"] = 0
                assigned.add(best_j)
                updates[tid] = self.tracks[tid]["box"]
            else:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
        for j, d in enumerate(dets):
            if j in assigned:
                continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = {"box": d[:4].tolist(), "age": 0}
            updates[tid] = self.tracks[tid]["box"]
        return [(tid, box) for tid, box in updates.items()]


class PPETracker:
    """Envolve ByteTrack/IoU e mantém estado de EPI com debounce por track_id."""
    def __init__(self, fps_hint: float = 0.5, debounce_seconds: float = 8.0,
                 track_thresh: float = 0.25, match_thresh: float = 0.3,
                 track_buffer: int = 60, iou_thresh: float = 0.3, max_age: int = 30):
        """
        Args:
            fps_hint: FPS aproximado do stream
            debounce_seconds: Tempo de debounce para violações de EPI
            track_thresh: Threshold mínimo de confiança para criar novo track (ByteTrack)
            match_thresh: Threshold de IoU para associar detecções a tracks existentes (ByteTrack)
                         Valores menores = mais permissivo = menos re-ID (recomendado: 0.3-0.5)
            track_buffer: Número de frames para manter track perdido antes de remover (ByteTrack)
                         Valores maiores = mais persistência (recomendado: 30-90)
            iou_thresh: Threshold de IoU para SimpleIoUTracker (fallback)
            max_age: Frames máximos sem detecção antes de remover track (SimpleIoUTracker)
        """
        self.fps_hint = fps_hint
        self.debounce = debounce_seconds
        self.track_engine = None
        self.use_byte = False

        if _HAS_ULT_BYTE:
            try:
                # Se track_buffer não foi especificado, calcula baseado no fps
                if track_buffer is None:
                    track_buffer = max(30, int(30 * fps_hint))  # Mínimo 30 frames
                
                args = type("Args", (), dict(
                    track_thresh=track_thresh,
                    match_thresh=match_thresh,  # Reduzido de 0.8 para 0.3 (mais permissivo)
                    track_buffer=track_buffer,  # Aumentado significativamente
                    mot20=False
                ))
                self.track_engine = BYTETracker(args, frame_rate=max(1, int(fps_hint)))
                self.use_byte = True
            except Exception:
                self.track_engine = SimpleIoUTracker(iou_thresh=iou_thresh, max_age=max_age)
        else:
            self.track_engine = SimpleIoUTracker(iou_thresh=iou_thresh, max_age=max_age)

        self.tr_states: Dict[int, Dict] = {}  # id -> {"last_seen": ts, "ppe":{}, "missing_since":{}, "frames": 0}

    def _byte_update(self, tlwhs: np.ndarray, scores: np.ndarray, frame_size: Optional[Tuple[int,int]]):
        """
        Tenta várias assinaturas do BYTETracker.update:
        - update(dets5, img_size, img_size)
        - update(dets5, img_size)
        - update(dets5)
        - update(tlwhs, scores, img_size)
        - update(tlwhs, scores)
        Retorna lista de alvos ou levanta Exception para fallback.
        """
        # dets5 = [tlwh + score]
        dets5 = np.hstack([tlwhs, scores.reshape(-1, 1)]) if tlwhs.size else tlwhs
        H = W = None
        if frame_size is not None:
            H, W = int(frame_size[0]), int(frame_size[1])

        tried = []
        try:
            if H is not None and W is not None:
                return self.track_engine.update(dets5, (H, W), (H, W))
            tried.append("dets5,(H,W),(H,W)")
        except TypeError:
            pass
        try:
            if H is not None and W is not None:
                return self.track_engine.update(dets5, (H, W))
            tried.append("dets5,(H,W)")
        except TypeError:
            pass
        try:
            return self.track_engine.update(dets5)
        except TypeError:
            pass
        try:
            if H is not None and W is not None:
                return self.track_engine.update(tlwhs, scores, (H, W))
            tried.append("tlwhs,scores,(H,W)")
        except TypeError:
            pass
        try:
            return self.track_engine.update(tlwhs, scores)
        except TypeError as e:
            raise RuntimeError(f"BYTETracker.update assinaturas falharam; tentativas={tried}") from e

    def _parse_online_targets(self, online_targets):
        """Normaliza a saída do ByteTrack para [{'id': int, 'box':[x1,y1,x2,y2]}]."""
        tracks = []
        try:
            for t in online_targets:
                if hasattr(t, "tlwh"):
                    x, y, w, h = t.tlwh
                    box = [float(x), float(y), float(x+w), float(y+h)]
                elif hasattr(t, "tlbr"):
                    x1, y1, x2, y2 = t.tlbr
                    box = [float(x1), float(y1), float(x2), float(y2)]
                elif isinstance(t, (list, tuple)) and len(t) >= 5:
                    # heurística: [x1,y1,x2,y2,id,...]
                    x1,y1,x2,y2,tid = t[:5]
                    box = [float(x1), float(y1), float(x2), float(y2)]
                    tracks.append({"id": int(tid), "box": box})
                    continue
                else:
                    # formato desconhecido
                    return None
                tid = int(getattr(t, "track_id", getattr(t, "id", -1)))
                if tid < 0:
                    return None
                tracks.append({"id": tid, "box": box})
            return tracks
        except Exception:
            return None

    def update(self, dets_xyxy_conf: np.ndarray, frame_size: Optional[Tuple[int,int]] = None):
        """
        dets_xyxy_conf: np.ndarray shape [N,5] -> x1,y1,x2,y2,conf
        frame_size: (H, W) opcional, usado por algumas versões do ByteTrack.
        Retorna: [{"id": id, "box":[x1,y1,x2,y2]}]
        """
        tracks = []
        if dets_xyxy_conf.size == 0:
            return tracks

        if self.use_byte:
            # monta tlwh + scores
            tlwhs, scores = [], []
            for d in dets_xyxy_conf:
                x1,y1,x2,y2,conf = d.tolist()
                tlwhs.append([x1, y1, x2-x1, y2-y1])
                scores.append(conf)
            tlwhs = np.array(tlwhs, dtype=float)
            scores = np.array(scores, dtype=float)
            try:
                online_targets = self._byte_update(tlwhs, scores, frame_size)
                parsed = self._parse_online_targets(online_targets)
                if parsed is None:
                    # Formato inesperado → fallback
                    self.use_byte = False
                    self.track_engine = SimpleIoUTracker()
                else:
                    tracks = parsed
            except Exception:
                # Qualquer erro → fallback
                self.use_byte = False
                self.track_engine = SimpleIoUTracker()

        if not self.use_byte:
            pairs = self.track_engine.update(dets_xyxy_conf)
            for tid, box in pairs:
                tracks.append({"id": int(tid), "box": list(map(float, box))})

        # atualiza estado por track
        now = time.time()
        for tr in tracks:
            tid = tr["id"]
            st = self.tr_states.setdefault(tid, {"last_seen": now, "ppe": {}, "missing_since": {}, "frames": 0})
            st["last_seen"] = now
            st["frames"] += 1
        return tracks

    def should_recheck(self, tid: int, every_n_frames: int = 5) -> bool:
        st = self.tr_states.get(tid)
        if not st:
            return True
        return (st["frames"] % max(1, every_n_frames)) == 0

    def update_ppe(self, tid: int, flags: Dict[str, bool], confidence_scores: Optional[Dict[str, float]] = None):
        """
        Atualiza flags EPI com debounce melhorado.
        Retorna True/False/'PENDING' por chave.
        confidence_scores: scores de confiança opcionais para ajustar debounce dinamicamente.
        """
        now = time.time()
        
        # Garante que o estado existe e tem todas as chaves necessárias
        if tid not in self.tr_states:
            self.tr_states[tid] = {
                "last_seen": now,
                "ppe": {},
                "missing_since": {},
                "present_since": {},
                "frames": 0,
                "confidence": {}
            }
        
        st = self.tr_states[tid]
        
        # Garante que todas as chaves necessárias existem (para estados antigos)
        if "present_since" not in st:
            st["present_since"] = {}
        if "confidence" not in st:
            st["confidence"] = {}
        if "missing_since" not in st:
            st["missing_since"] = {}
        if "ppe" not in st:
            st["ppe"] = {}
        
        st["last_seen"] = now
        out = {}
        confidence_scores = confidence_scores or {}
        
        for k, v in flags.items():
            conf = confidence_scores.get(k, 0.5)  # Confiança padrão
            
            if v:
                # EPI detectado
                if k not in st["present_since"]:
                    st["present_since"][k] = now
                st["missing_since"].pop(k, None)
                st["confidence"][k] = conf
                
                # Debounce mais rápido para alta confiança
                present_duration = now - st["present_since"].get(k, now)
                # Alta confiança (>0.7): confirma em 1s, média (0.5-0.7): 2s, baixa (<0.5): 3s
                confirm_threshold = 1.0 if conf > 0.7 else (2.0 if conf > 0.5 else 3.0)
                
                if present_duration >= confirm_threshold or st["ppe"].get(k) is True:
                    st["ppe"][k] = True
                    out[k] = True
                else:
                    out[k] = "PENDING"
            else:
                # EPI não detectado
                if k not in st["missing_since"]:
                    st["missing_since"][k] = now
                st["present_since"].pop(k, None)
                
                # Debounce adaptativo: mais rápido para remover se nunca foi confirmado
                missing_duration = now - st["missing_since"].get(k, now)
                was_confirmed = st["ppe"].get(k) is True
                
                # Se nunca foi confirmado, remove mais rápido (metade do debounce)
                # Se foi confirmado, usa debounce completo
                remove_threshold = self.debounce * 0.5 if not was_confirmed else self.debounce
                
                if missing_duration >= remove_threshold:
                    st["ppe"][k] = False
                    out[k] = False
                else:
                    out[k] = "PENDING" if was_confirmed else False  # Se nunca confirmado, já marca como False
        return out
