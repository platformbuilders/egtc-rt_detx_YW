# alerts.py - Sistema de alertas com persistência em banco e supressão via Redis
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import io
import base64

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[WARN] requests não disponível. Alertas Telegram não funcionarão.")

try:
    from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.exc import OperationalError
    Base = declarative_base()
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Base = None
    create_engine = None
    sessionmaker = None
    Column = None
    Integer = None
    String = None
    DateTime = None
    Float = None
    Text = None
    Boolean = None
    OperationalError = None
    print("[WARN] SQLAlchemy não disponível. Alertas não serão persistidos em banco.")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("[WARN] Redis não disponível. Supressão de alertas não funcionará.")

# Define PPEAlert apenas se SQLAlchemy estiver disponível
if SQLALCHEMY_AVAILABLE and Base is not None:
    class PPEAlert(Base):
        """Modelo de dados para alertas de EPI"""
        __tablename__ = 'ppe_alerts'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        camera_id = Column(String(50), nullable=False, index=True)
        timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
        grid_x = Column(Integer, nullable=False)
        grid_y = Column(Integer, nullable=False)
        person_track_id = Column(Integer, nullable=True)
        missing_ppe = Column(Text, nullable=False)  # JSON array de EPIs faltando
        person_box = Column(Text, nullable=True)  # JSON array [x1, y1, x2, y2]
        roi_name = Column(String(100), nullable=True)
        frame_width = Column(Integer, nullable=True)
        frame_height = Column(Integer, nullable=True)
        confidence_scores = Column(Text, nullable=True)  # JSON dict de scores
        alert_duration_seconds = Column(Float, nullable=True)  # Tempo que a violação persistiu antes do alerta
        image_path = Column(String(500), nullable=True)  # Caminho da imagem salva (crop ou frame completo)
else:
    # Classe dummy se SQLAlchemy não estiver disponível
    class PPEAlert:
        pass

@dataclass
class AlertConfig:
    """Configuração do sistema de alertas"""
    db_type: str = "mysql"  # mysql, postgresql, oracle
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = "sql_user"
    db_password: str = "Passw0rd"
    db_name: str = "egtc_alerts"
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    alert_debounce_seconds: float = 15.0  # Tempo para confirmar violação antes de alertar (aumentado para reduzir falsos positivos)
    alert_min_consecutive_frames: int = 20  # Frames consecutivos mínimos da MESMA PESSOA (track_id) sem EPI
    suppression_reset_seconds: float = 20.0  # Tempo sem violação para resetar supressão (reduzido para 20s)
    alert_hash_ttl_seconds: float = 60.0  # TTL do hash de violação (renovado enquanto violação persistir)
    grid_size: int = 8  # Grid 8x8
    grid_cell_size_meters: float = 2.0  # 2 metros por célula
    
    # Telegram
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    telegram_enabled: bool = False
    timezone_offset_hours: float = -3.0  # Offset do timezone local (GMT-3 = -3.0)
    
    # Salvamento de imagens
    save_alert_images: bool = True  # Se True, salva imagens de alertas
    save_crop_only: bool = True  # Se True, salva apenas crop da pessoa; se False, salva frame completo
    crops_dir: str = "crops"  # Diretório para salvar crops/frames
    
    @classmethod
    def from_env_file(cls, env_path: str = "db_config.env") -> 'AlertConfig':
        """Carrega configuração de arquivo .env"""
        config = cls()
        
        if not os.path.exists(env_path):
            print(f"[WARN] Arquivo de configuração {env_path} não encontrado. Usando valores padrão.")
            return config
        
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key == "DB_TYPE":
                            config.db_type = value
                        elif key == "DB_HOST":
                            config.db_host = value
                        elif key == "DB_PORT":
                            config.db_port = int(value)
                        elif key == "DB_USER":
                            config.db_user = value
                        elif key == "DB_PASSWORD":
                            config.db_password = value
                        elif key == "DB_NAME":
                            config.db_name = value
                        elif key == "REDIS_HOST":
                            config.redis_host = value
                        elif key == "REDIS_PORT":
                            config.redis_port = int(value)
                        elif key == "REDIS_DB":
                            config.redis_db = int(value)
                        elif key == "REDIS_PASSWORD":
                            config.redis_password = value if value else None
                        elif key == "TELEGRAM_TOKEN":
                            config.telegram_token = value
                            config.telegram_enabled = True
                        elif key == "TELEGRAM_CHAT_ID":
                            config.telegram_chat_id = value
                            config.telegram_enabled = True
                        elif key == "TIMEZONE_OFFSET_HOURS":
                            config.timezone_offset_hours = float(value)
                        elif key == "SAVE_ALERT_IMAGES":
                            config.save_alert_images = value.lower() in ("true", "1", "yes")
                        elif key == "SAVE_CROP_ONLY":
                            config.save_crop_only = value.lower() in ("true", "1", "yes")
                        elif key == "CROPS_DIR":
                            config.crops_dir = value
            
            if config.telegram_token and config.telegram_chat_id:
                config.telegram_enabled = True
                print(f"[INFO] Telegram habilitado para chat {config.telegram_chat_id}")
            
            print(f"[INFO] Timezone configurado: GMT{config.timezone_offset_hours:+.1f}")
            if config.save_alert_images:
                print(f"[INFO] Salvamento de imagens: {'Crop apenas' if config.save_crop_only else 'Frame completo'} em '{config.crops_dir}'")
            
            print(f"[INFO] Configuração de banco carregada de {env_path}")
        except Exception as e:
            print(f"[ERROR] Erro ao carregar configuração de {env_path}: {e}")
        
        return config

class AlertManager:
    """Gerenciador de alertas com persistência e supressão"""
    
    def __init__(self, config: AlertConfig, camera_id: str, frame_width: int, frame_height: int, send_alerts: bool = True):
        self.config = config
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.send_alerts = send_alerts  # Controla se alertas são enviados (banco/Telegram) ou apenas processados (para grid)
        self.violation_start_times: Dict[Tuple[int, int], float] = {}  # Para rastrear tempo de violação
        self.frame_counter = 0  # Contador de frames para rastrear frames consecutivos
        
        # Calcula tamanho de cada célula do grid em pixels
        self.cell_width = frame_width / config.grid_size
        self.cell_height = frame_height / config.grid_size
        
        # Cria diretório de crops se necessário
        if config.save_alert_images:
            crops_path = os.path.abspath(config.crops_dir)
            os.makedirs(crops_path, exist_ok=True)
            print(f"[INFO] Diretório de crops: {crops_path}")
        
        # Inicializa banco de dados (apenas se send_alerts=True)
        self.db_session = None
        if self.send_alerts and SQLALCHEMY_AVAILABLE:
            self._init_database()
        
        # Inicializa Redis (sempre necessário para supressão, mesmo sem envio)
        self.redis_client = None
        if REDIS_AVAILABLE:
            self._init_redis()
        
        # Estado interno: violações por track_id (pessoa)
        # {track_id: {"violation_start": timestamp, "first_frame": int, "consecutive_frames": int, 
        #             "missing_ppe": [...], "last_seen": timestamp, "last_seen_frame": int, 
        #             "grid_x": int, "grid_y": int, "boxes": [...], "alerted": bool}}
        self.violation_states: Dict[int, Dict] = {}
    
    def _init_database(self):
        """Inicializa conexão com banco de dados"""
        if not SQLALCHEMY_AVAILABLE:
            print("[WARN] SQLAlchemy não disponível. Banco de dados não será inicializado.")
            self.db_session = None
            return
        
        try:
            # Monta URL de conexão baseado no tipo de banco
            if self.config.db_type == "mysql":
                # Para MySQL, tenta criar banco se não existir
                try:
                    import pymysql
                    # Conecta sem especificar banco para criar se necessário
                    temp_conn = pymysql.connect(
                        host=self.config.db_host,
                        port=self.config.db_port,
                        user=self.config.db_user,
                        password=self.config.db_password,
                        charset='utf8mb4'
                    )
                    with temp_conn.cursor() as cursor:
                        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.config.db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                    temp_conn.close()
                    print(f"[INFO] Banco de dados {self.config.db_name} verificado/criado")
                except ImportError:
                    print(f"[WARN] pymysql não disponível. Instale: pip install pymysql")
                except Exception as e:
                    print(f"[WARN] Não foi possível criar banco {self.config.db_name}: {e}")
                    print(f"[INFO] Tentando conectar ao banco existente...")
                
                db_url = f"mysql+pymysql://{self.config.db_user}:{self.config.db_password}@{self.config.db_host}:{self.config.db_port}/{self.config.db_name}?charset=utf8mb4"
            elif self.config.db_type == "postgresql":
                db_url = f"postgresql://{self.config.db_user}:{self.config.db_password}@{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
            elif self.config.db_type == "oracle":
                db_url = f"oracle+cx_oracle://{self.config.db_user}:{self.config.db_password}@{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
            else:
                print(f"[ERROR] Tipo de banco não suportado: {self.config.db_type}")
                return
            
            engine = create_engine(db_url, echo=False)
            
            # Tenta criar tabela se não existir
            try:
                if Base is not None:
                    Base.metadata.create_all(engine)
                    print(f"[INFO] Tabela de alertas verificada/criada no banco {self.config.db_type}")
                else:
                    print(f"[WARN] Base não disponível. Tabela não será criada.")
            except Exception as e:
                print(f"[WARN] Erro ao criar tabela (pode já existir): {e}")
            
            # Verifica e adiciona coluna image_path se não existir
            self._ensure_image_path_column(engine)
            
            Session = sessionmaker(bind=engine)
            self.db_session = Session()
        except Exception as e:
            print(f"[ERROR] Falha ao conectar ao banco de dados: {e}")
            self.db_session = None
    
    def _ensure_image_path_column(self, engine):
        """Verifica se a coluna image_path existe e adiciona se necessário"""
        try:
            from sqlalchemy import inspect, text
            
            inspector = inspect(engine)
            columns = [col['name'] for col in inspector.get_columns('ppe_alerts')]
            
            if 'image_path' not in columns:
                print(f"[INFO] Adicionando coluna 'image_path' à tabela 'ppe_alerts'...")
                with engine.begin() as conn:  # begin() faz commit automático
                    if self.config.db_type == "mysql":
                        conn.execute(text("ALTER TABLE ppe_alerts ADD COLUMN image_path VARCHAR(500) NULL"))
                    elif self.config.db_type == "postgresql":
                        conn.execute(text("ALTER TABLE ppe_alerts ADD COLUMN image_path VARCHAR(500)"))
                    elif self.config.db_type == "oracle":
                        conn.execute(text("ALTER TABLE ppe_alerts ADD image_path VARCHAR2(500)"))
                print(f"[INFO] Coluna 'image_path' adicionada com sucesso")
            else:
                print(f"[INFO] Coluna 'image_path' já existe na tabela")
        except Exception as e:
            print(f"[WARN] Não foi possível verificar/adicionar coluna 'image_path': {e}")
            import traceback
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            # Não falha completamente, apenas avisa
    
    def _init_redis(self):
        """Inicializa conexão com Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True
            )
            # Testa conexão
            self.redis_client.ping()
            print(f"[INFO] Conectado ao Redis em {self.config.redis_host}:{self.config.redis_port}")
        except Exception as e:
            print(f"[ERROR] Falha ao conectar ao Redis: {e}")
            self.redis_client = None
    
    def _get_grid_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Converte coordenada (x, y) em pixels para célula do grid (grid_x, grid_y)"""
        grid_x = min(int(x / self.cell_width), self.config.grid_size - 1)
        grid_y = min(int(y / self.frame_height * self.config.grid_size), self.config.grid_size - 1)
        return (grid_x, grid_y)
    
    def _get_violation_hash(self, grid_x: int, grid_y: int, missing_ppe: List[str]) -> str:
        """Gera hash único para uma violação (posição + EPIs faltando)"""
        # Ordena EPIs para garantir hash consistente
        ppe_sorted = tuple(sorted(missing_ppe))
        return f"{self.camera_id}:{grid_x}:{grid_y}:{ppe_sorted}"
    
    def _is_violation_hash_alerted(self, violation_hash: str) -> bool:
        """Verifica se uma violação (hash) já foi alertada recentemente"""
        if not self.redis_client:
            return False
        key = f"violation_hash:{violation_hash}"
        return self.redis_client.exists(key) > 0
    
    def _mark_violation_hash_alerted(self, violation_hash: str):
        """Marca uma violação (hash) como alertada"""
        if not self.redis_client:
            return
        key = f"violation_hash:{violation_hash}"
        # TTL baseado em alert_hash_ttl_seconds (60s)
        self.redis_client.setex(key, int(self.config.alert_hash_ttl_seconds), "1")
    
    def _is_suppressed(self, grid_x: int, grid_y: int) -> bool:
        """Verifica se a célula está com supressão ativa"""
        if not self.redis_client:
            return False
        
        key = f"alert_suppression:{self.camera_id}:{grid_x}:{grid_y}"
        return self.redis_client.exists(key) > 0
    
    def _set_suppression(self, grid_x: int, grid_y: int, renew: bool = False):
        """
        Marca célula como suprimida.
        
        Args:
            grid_x, grid_y: Coordenadas da célula
            renew: Se True, renova o TTL se já existe (para supressão permanente enquanto violação persistir)
        """
        if not self.redis_client:
            return
        
        key = f"alert_suppression:{self.camera_id}:{grid_x}:{grid_y}"
        if renew and self.redis_client.exists(key):
            # Renova TTL para manter supressão enquanto violação persistir
            self.redis_client.expire(key, int(self.config.suppression_reset_seconds))
        else:
            # Define TTL para reset automático após suppression_reset_seconds
            self.redis_client.setex(key, int(self.config.suppression_reset_seconds), "1")
    
    def _clear_suppression(self, grid_x: int, grid_y: int):
        """Remove supressão de uma célula"""
        if not self.redis_client:
            return
        
        key = f"alert_suppression:{self.camera_id}:{grid_x}:{grid_y}"
        self.redis_client.delete(key)
    
    def update_violations(self, violations: List[Tuple[int, List[float], List[str]]]):
        """
        Atualiza estado de violações por track_id (pessoa).
        
        NOVA LÓGICA: Rastreia violações por pessoa (track_id), não por célula.
        Exige que a MESMA pessoa apareça X frames consecutivos sem EPI.
        
        Args:
            violations: Lista de (track_id, [x1, y1, x2, y2], [missing_ppe_items])
        """
        now = time.time()
        self.frame_counter += 1  # Incrementa contador de frames
        
        # Identifica track_ids ativos nesta frame
        # violations pode ter 3 ou 4 elementos: (tid, box, missing_ppe) ou (tid, box, missing_ppe, roi_name)
        active_track_ids = {violation[0] for violation in violations}
        
        # Remove violações de pessoas que não aparecem mais nesta frame
        # Se uma pessoa não aparece, RESETA seu contador (força confirmação contínua)
        track_ids_to_remove = []
        for track_id in list(self.violation_states.keys()):
            if track_id not in active_track_ids:
                state = self.violation_states[track_id]
                # Se já foi alertada, mantém por um tempo para evitar re-alerta
                if state.get("alerted", False):
                    time_since_last_seen = now - state.get("last_seen", now)
                    if time_since_last_seen > self.config.suppression_reset_seconds:
                        track_ids_to_remove.append(track_id)
                else:
                    # Não foi alertada ainda - RESETA imediatamente (força confirmação contínua)
                    track_ids_to_remove.append(track_id)
                    print(f"[DEBUG ALERT] {self.camera_id}: Pessoa {track_id} não apareceu neste frame - RESETANDO contagem de frames consecutivos")
        
        for track_id in track_ids_to_remove:
            if track_id in self.violation_states:
                del self.violation_states[track_id]
        
        # Atualiza estados de violações ativas (por track_id)
        # violations pode ser: (tid, box, missing_ppe) ou (tid, box, missing_ppe, roi_name)
        for violation in violations:
            if len(violation) == 3:
                tid, box, missing_ppe = violation
                roi_name = None
            else:
                tid, box, missing_ppe, roi_name = violation
            
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            grid_x, grid_y = self._get_grid_cell(cx, cy)
            
            if tid not in self.violation_states:
                # Nova violação para esta pessoa - inicia contagem
                self.violation_states[tid] = {
                    "violation_start": now,
                    "first_frame": self.frame_counter,
                    "last_seen": now,
                    "last_seen_frame": self.frame_counter,
                    "consecutive_frames": 1,  # Primeiro frame consecutivo
                    "missing_ppe": missing_ppe.copy(),
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "boxes": [box.copy()],
                    "roi_name": roi_name,  # Armazena nome do ROI (pode ser None)
                    "alerted": False
                }
                print(f"[DEBUG ALERT] {self.camera_id}: Nova violação detectada para pessoa {tid} - EPIs faltando: {missing_ppe} - ROI: {roi_name} - Frame {self.frame_counter}")
            else:
                # Pessoa já está sendo rastreada - incrementa frames consecutivos
                state = self.violation_states[tid]
                # Verifica se é frame consecutivo (deve ser o próximo frame após o último visto)
                if self.frame_counter == state["last_seen_frame"] + 1:
                    # Frame consecutivo - incrementa contador
                    state["consecutive_frames"] += 1
                else:
                    # Não é frame consecutivo - RESETA contador (pessoa sumiu e voltou)
                    print(f"[DEBUG ALERT] {self.camera_id}: Pessoa {tid} voltou após {self.frame_counter - state['last_seen_frame']} frames - RESETANDO contagem")
                    state["consecutive_frames"] = 1
                    state["first_frame"] = self.frame_counter
                    state["violation_start"] = now
                
                state["last_seen"] = now
                state["last_seen_frame"] = self.frame_counter
                # Atualiza lista de EPIs faltando (união)
                state["missing_ppe"] = list(set(state["missing_ppe"] + missing_ppe))
                # Atualiza posição do grid
                state["grid_x"] = grid_x
                state["grid_y"] = grid_y
                # Atualiza ROI (usa o mais recente se disponível, mas preserva se já existir e novo for None)
                if roi_name:
                    state["roi_name"] = roi_name
                # Se ROI não foi definido ainda e temos um novo, define
                elif "roi_name" not in state or state.get("roi_name") is None:
                    # Mantém None se não houver ROI
                    pass
                # Mantém box mais recente
                if box not in state["boxes"]:
                    state["boxes"].append(box.copy())
    
    def check_and_generate_alerts(self, frame_bgr: Optional = None) -> List[Dict]:
        """
        Verifica violações confirmadas e gera alertas se necessário.
        
        NOVA LÓGICA:
        - Rastreia por track_id (pessoa), não por célula
        - Exige X frames consecutivos da MESMA pessoa sem EPI
        - Reset automático se pessoa não aparecer em um frame
        - Hash de violação: evita alertas duplicados
        
        Args:
            frame_bgr: Frame BGR opcional para extrair crop da pessoa
        
        Returns:
            Lista de alertas gerados (cada um é um dict com informações do alerta)
        """
        now = time.time()
        alerts_generated = []
        
        for track_id, state in list(self.violation_states.items()):
            # Ignora se já foi alertada (mantém no estado para evitar re-alerta)
            if state.get("alerted", False):
                continue
            
            violation_duration = now - state["violation_start"]
            consecutive_frames = state.get("consecutive_frames", 0)
            grid_x = state["grid_x"]
            grid_y = state["grid_y"]
            
            # NOVA LÓGICA: Exige frames consecutivos E tempo mínimo (ambos devem ser satisfeitos)
            # Isso garante que a pessoa realmente está sem EPI de forma consistente
            frames_ok = consecutive_frames >= self.config.alert_min_consecutive_frames
            time_ok = violation_duration >= self.config.alert_debounce_seconds
            is_confirmed = frames_ok and time_ok  # Lógica E (mais restritiva)
            
            # DEBUG: Log para verificar condições
            if violation_duration > 0 and consecutive_frames > 1:
                print(f"[DEBUG ALERT] {self.camera_id} Pessoa {track_id} (célula {grid_x},{grid_y}): "
                      f"frames={consecutive_frames}/{self.config.alert_min_consecutive_frames}, "
                      f"tempo={violation_duration:.2f}s/{self.config.alert_debounce_seconds}s, "
                      f"confirmado={is_confirmed}, EPIs={state['missing_ppe']}")
            
            # Verifica se violação está confirmada (frames E tempo)
            if is_confirmed:
                # Gera hash da violação para evitar duplicatas
                violation_hash = self._get_violation_hash(grid_x, grid_y, state["missing_ppe"])
                
                # Verifica se já foi alertada recentemente (hash)
                if self._is_violation_hash_alerted(violation_hash):
                    print(f"[DEBUG ALERT] {self.camera_id} Pessoa {track_id}: violação já foi alertada (hash: {violation_hash[:50]}...) - ignorando")
                    # Renova supressão enquanto violação persistir
                    if self._is_suppressed(grid_x, grid_y):
                        self._set_suppression(grid_x, grid_y, renew=True)
                    # Marca como alertada mas mantém no estado
                    state["alerted"] = True
                    continue
                
                # Verifica se célula não está suprimida
                if not self._is_suppressed(grid_x, grid_y):
                    print(f"[ALERT] {self.camera_id}: Gerando alerta para pessoa {track_id} na célula ({grid_x}, {grid_y}) - "
                          f"frames={consecutive_frames}, tempo={violation_duration:.2f}s, EPIs={state['missing_ppe']}")
                    
                    # Gera alerta (passa frame para Telegram)
                    alert = self._create_alert(grid_x, grid_y, state, violation_duration, frame_bgr=frame_bgr, track_id=track_id)
                    alerts_generated.append(alert)
                    
                    # Marca hash da violação como alertada
                    self._mark_violation_hash_alerted(violation_hash)
                    
                    # Marca célula como suprimida
                    self._set_suppression(grid_x, grid_y)
                    
                    # Marca como alertada mas mantém no estado para evitar re-alerta
                    state["alerted"] = True
                else:
                    # Célula está suprimida - renova supressão enquanto violação persistir
                    self._set_suppression(grid_x, grid_y, renew=True)
                    state["alerted"] = True
                    print(f"[DEBUG ALERT] {self.camera_id} Pessoa {track_id}: violação confirmada mas célula está suprimida (renovando)")
        
        return alerts_generated
    
    def _save_alert_image(self, frame_bgr, box: Optional[List[float]] = None) -> Optional[str]:
        """
        Salva imagem do alerta (crop ou frame completo).
        
        Args:
            frame_bgr: Frame BGR completo
            box: Bounding box [x1, y1, x2, y2] se save_crop_only=True
            
        Returns:
            Caminho relativo da imagem salva ou None se não salvou
        """
        if not self.config.save_alert_images or frame_bgr is None:
            return None
        
        try:
            import cv2
            import numpy as np
            
            # Gera nome do arquivo baseado em timestamp e câmera
            timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milissegundos
            filename = f"{self.camera_id}_{timestamp_str}.jpg"
            filepath = os.path.join(self.config.crops_dir, filename)
            filepath_abs = os.path.abspath(filepath)
            
            # Salva crop ou frame completo
            if self.config.save_crop_only and box is not None:
                # Extrai crop da pessoa
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_bgr.shape[1], x2)
                y2 = min(frame_bgr.shape[0], y2)
                
                # Adiciona padding ao crop
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame_bgr.shape[1], x2 + padding)
                y2 = min(frame_bgr.shape[0], y2 + padding)
                
                crop = frame_bgr[y1:y2, x1:x2]
                
                if crop.size == 0:
                    print(f"[WARN] Crop vazio, salvando frame completo")
                    image_to_save = frame_bgr
                else:
                    image_to_save = crop
            else:
                # Salva frame completo
                image_to_save = frame_bgr
            
            # Salva imagem
            cv2.imwrite(filepath_abs, image_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Retorna caminho relativo
            return filepath
            
        except Exception as e:
            print(f"[ERROR] Falha ao salvar imagem do alerta: {e}")
            return None
    
    def _create_alert(self, grid_x: int, grid_y: int, state: Dict, duration: float, 
                     frame_bgr: Optional = None, track_id: Optional[int] = None) -> Dict:
        """Cria e persiste um alerta"""
        # Pega box mais recente
        box = state["boxes"][-1] if state["boxes"] else None
        # Pega track_id (pessoa) - agora state é por track_id, então track_id é passado como parâmetro
        if track_id is None and "track_ids" in state:
            # Formato antigo (por célula) - pega primeiro track_id
            track_id = list(state["track_ids"])[0] if state["track_ids"] else None
        
        # Salva imagem se configurado
        image_path = None
        if self.config.save_alert_images and frame_bgr is not None:
            image_path = self._save_alert_image(frame_bgr, box if self.config.save_crop_only else None)
        
        alert_data = {
            "camera_id": self.camera_id,
            "timestamp": datetime.utcnow(),
            "grid_x": grid_x,
            "grid_y": grid_y,
            "person_track_id": track_id if track_id is not None else (list(state["track_ids"])[0] if "track_ids" in state and state["track_ids"] else None),
            "missing_ppe": json.dumps(state["missing_ppe"]),
            "person_box": json.dumps(box) if box else None,
            "roi_name": state.get("roi_name", None),  # Nome do ROI onde ocorreu a violação
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "confidence_scores": None,  # Pode ser preenchido depois
            "alert_duration_seconds": duration,
            "image_path": image_path
        }
        
        # Persiste no banco (apenas se send_alerts=True)
        if self.send_alerts and self.db_session:
            try:
                alert = PPEAlert(**alert_data)
                self.db_session.add(alert)
                self.db_session.commit()
                print(f"[ALERT] {self.camera_id}: Alerta gerado e persistido na célula ({grid_x}, {grid_y}) - EPIs faltando: {state['missing_ppe']}")
            except Exception as e:
                print(f"[ERROR] Falha ao persistir alerta no banco: {e}")
                self.db_session.rollback()
        elif not self.send_alerts:
            print(f"[ALERT] {self.camera_id}: Alerta detectado na célula ({grid_x}, {grid_y}) - EPIs faltando: {state['missing_ppe']} (não enviado - send_alerts=False)")
        
        # Envia para Telegram se habilitado (apenas se send_alerts=True)
        if self.send_alerts and self.config.telegram_enabled and frame_bgr is not None and box is not None:
            self._send_telegram_alert(alert_data, state, duration, frame_bgr, box)
        
        return alert_data
    
    def _send_telegram_alert(self, alert_data: Dict, state: Dict, duration: float, 
                            frame_bgr, box: List[float]):
        """Envia alerta para Telegram com crop da pessoa e mensagem formatada"""
        if not REQUESTS_AVAILABLE or not self.config.telegram_token or not self.config.telegram_chat_id:
            return
        
        try:
            import cv2
            import numpy as np
            
            # Extrai crop da pessoa
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_bgr.shape[1], x2)
            y2 = min(frame_bgr.shape[0], y2)
            
            # Adiciona padding ao crop
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame_bgr.shape[1], x2 + padding)
            y2 = min(frame_bgr.shape[0], y2 + padding)
            
            crop = frame_bgr[y1:y2, x1:x2]
            
            if crop.size == 0:
                print("[WARN] Crop vazio, não enviando para Telegram")
                return
            
            # Converte para JPEG em memória
            _, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            crop_bytes = buffer.tobytes()
            
            # Formata mensagem
            missing_ppe_list = state["missing_ppe"]
            missing_ppe_text = "\n".join([f"  • {item}" for item in missing_ppe_list])
            
            # Traduz EPIs para português
            ppe_translations = {
                "helmet": "Capacete",
                "gloves": "Luvas",
                "ear_protection": "Protetor Auricular",
                "vest": "Colete Refletivo",
                "apron": "Avental",
                "helmet_white": "Capacete Branco",
                "helmet_red": "Capacete Vermelho",
                "helmet_blue": "Capacete Azul",
                "helmet_yellow": "Capacete Amarelo",
                "helmet_brown": "Capacete Marrom",
            }
            
            missing_ppe_translated = []
            for item in missing_ppe_list:
                translated = ppe_translations.get(item, item)
                missing_ppe_translated.append(translated)
            
            missing_ppe_text = "\n".join([f"  • {item}" for item in missing_ppe_translated])
            
            # Formata data e hora no timezone local
            dt = alert_data["timestamp"]
            if isinstance(dt, datetime):
                # Se datetime não tem timezone, assume UTC
                if dt.tzinfo is None:
                    dt_utc = dt.replace(tzinfo=timezone.utc)
                else:
                    dt_utc = dt
                # Converte para timezone local
                local_tz = timezone(timedelta(hours=self.config.timezone_offset_hours))
                dt_local = dt_utc.astimezone(local_tz)
                date_str = dt_local.strftime("%d/%m/%Y")
                time_str = dt_local.strftime("%H:%M:%S")
            else:
                # Fallback: usa horário local atual
                local_tz = timezone(timedelta(hours=self.config.timezone_offset_hours))
                dt_local = datetime.now(local_tz)
                date_str = dt_local.strftime("%d/%m/%Y")
                time_str = dt_local.strftime("%H:%M:%S")
            
            # Formata duração
            duration_min = int(duration // 60)
            duration_sec = int(duration % 60)
            if duration_min > 0:
                duration_str = f"{duration_min} minuto(s) e {duration_sec} segundo(s)"
            else:
                duration_str = f"{duration_sec} segundo(s)"
            
            # Obtém nome do ROI do estado
            roi_name = state.get("roi_name", None)
            print(f"[DEBUG TELEGRAM] {self.camera_id}: Enviando alerta para pessoa {track_id} - ROI do estado: '{roi_name}' (tipo: {type(roi_name)})")
            print(f"[DEBUG TELEGRAM] {self.camera_id}: Chaves do estado: {list(state.keys())}")
            if roi_name:
                # Se houver múltiplos ROIs separados por vírgula, formata melhor
                roi_list = [r.strip() for r in str(roi_name).split(",")] if "," in str(roi_name) else [str(roi_name)]
                roi_text = f"📍 *ROI:* {', '.join(roi_list)}"
                print(f"[DEBUG TELEGRAM] {self.camera_id}: ROI formatado: {roi_text}")
            else:
                roi_text = ""
                print(f"[DEBUG TELEGRAM] {self.camera_id}: ROI não encontrado no estado (None ou vazio)")
            
            # Monta mensagem formatada
            message_lines = [
                "🚨 *ALERTA DE VIOLAÇÃO DE EPI*",
                "",
                f"📹 *Câmera:* {self.camera_id}",
                f"📅 *Data:* {date_str}",
                f"🕐 *Hora:* {time_str}",
            ]
            if roi_text:
                message_lines.append(roi_text)
            message_lines.extend([
                f"📍 *Localização:* Célula ({alert_data['grid_x']}, {alert_data['grid_y']})",
                f"⏱️ *Tempo sem EPI:* {duration_str}",
                "",
                "❌ *EPIs Faltando:*",
                missing_ppe_text,
                "",
                f"🆔 *ID de Rastreamento:* {alert_data.get('person_track_id', 'N/A')}"
            ])
            message = "\n".join(message_lines)
            
            # Envia foto e mensagem
            url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendPhoto"
            
            files = {
                'photo': ('crop.jpg', crop_bytes, 'image/jpeg')
            }
            
            data = {
                'chat_id': self.config.telegram_chat_id,
                'caption': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, files=files, data=data, timeout=10)
            response.raise_for_status()
            
            print(f"[TELEGRAM] Alerta enviado para chat {self.config.telegram_chat_id}")
            
        except Exception as e:
            print(f"[ERROR] Falha ao enviar alerta para Telegram: {e}")
    
    def get_suppressed_cells(self) -> List[Tuple[int, int]]:
        """Retorna lista de células com supressão ativa"""
        if not self.redis_client:
            return []
        
        suppressed = []
        try:
            pattern = f"alert_suppression:{self.camera_id}:*"
            keys = self.redis_client.keys(pattern)
            
            for key in keys:
                # Formato da chave: alert_suppression:{camera_id}:{grid_x}:{grid_y}
                parts = key.split(":")
                if len(parts) >= 4:
                    try:
                        grid_x = int(parts[2])
                        grid_y = int(parts[3])
                        suppressed.append((grid_x, grid_y))
                    except (ValueError, IndexError) as e:
                        print(f"[WARN] Erro ao parsear chave Redis '{key}': {e}")
                        continue
        except Exception as e:
            print(f"[ERROR] Erro ao obter células suprimidas do Redis: {e}")
        
        return suppressed
    
    def get_alert_status(self, x: float, y: float, track_id: Optional[int] = None) -> Optional[str]:
        """
        Retorna o status do alerta para uma posição (x, y) em pixels ou track_id.
        
        Args:
            x, y: Coordenadas em pixels (centro do bounding box)
            track_id: ID de rastreamento da pessoa (opcional, mais preciso)
        
        Returns:
            "ALERTA GERADO" - Alerta foi gerado e enviado (célula suprimida)
            "AGUARDANDO CONFIRMAÇÃO" - Violação detectada mas ainda não confirmada (debounce)
            "VIOLAÇÃO ATIVA" - Violação confirmada mas ainda não alertada
            None - Sem violação ou alerta
        """
        grid_x, grid_y = self._get_grid_cell(x, y)
        
        # Verifica se célula está suprimida (alerta já gerado)
        if self._is_suppressed(grid_x, grid_y):
            return "ALERTA GERADO"
        
        # Busca violação por track_id (mais preciso) ou por posição
        state = None
        if track_id is not None and track_id in self.violation_states:
            state = self.violation_states[track_id]
            # Verifica se está na mesma célula (pode ter se movido)
            if state["grid_x"] == grid_x and state["grid_y"] == grid_y:
                pass  # Usa este state
            else:
                state = None  # Pessoa se moveu para outra célula
        
        # Se não encontrou por track_id, busca por posição (grid)
        if state is None:
            for tid, st in self.violation_states.items():
                if st["grid_x"] == grid_x and st["grid_y"] == grid_y:
                    state = st
                    break
        
        if state:
            now = time.time()
            violation_duration = now - state["violation_start"]
            consecutive_frames = state.get("consecutive_frames", 0)
            
            # Verifica condições de confirmação
            frames_ok = consecutive_frames >= self.config.alert_min_consecutive_frames
            time_ok = violation_duration >= self.config.alert_debounce_seconds
            is_confirmed = frames_ok and time_ok
            
            if is_confirmed:
                return "VIOLAÇÃO ATIVA"
            else:
                # Violação ainda em confirmação
                remaining_frames = max(0, self.config.alert_min_consecutive_frames - consecutive_frames)
                remaining_time = max(0, self.config.alert_debounce_seconds - violation_duration)
                if remaining_frames > 0:
                    return f"AGUARDANDO ({remaining_frames} frames)"
                else:
                    return f"AGUARDANDO ({int(remaining_time)}s)"
        
        return None
    
    def cleanup(self):
        """Limpa recursos"""
        if self.db_session:
            self.db_session.close()
        if self.redis_client:
            self.redis_client.close()

