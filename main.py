"""
Main - Processo principal que carrega modelos e gerencia threads de câmeras.
Arquitetura baseada em threading para processamento paralelo de múltiplas câmeras.
"""
import os
import sys
import time
import signal
import argparse
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml
import cv2
import numpy as np
import pickle

# Adiciona diretórios ao path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
egtc_olm_dir = os.path.join(os.path.dirname(script_dir), 'egtc_olm')
if os.path.exists(egtc_olm_dir):
    sys.path.append(egtc_olm_dir)

from rtdetr_detector import RTDETRPerson
from ppe_detector import UnifiedPPEDetector
from camera_thread import CameraThread
from stream_server import StreamServer
from logger import CentralizedLogger, set_logger, get_logger
from pipeline_RETDETRX_YW import load_yaml, read_prompts


class PipelineManager:
    """
    Gerencia o pipeline completo: carrega modelos, inicia threads de câmeras e servidor de streaming.
    """
    
    def __init__(self, config_path: str, prompt_path: str, alert_config_path: str = "db_config.env"):
        """
        Inicializa o gerenciador do pipeline.
        
        Args:
            config_path: Caminho do arquivo YAML de configuração
            prompt_path: Caminho do arquivo YAML de prompts de EPIs
            alert_config_path: Caminho do arquivo de configuração de alertas (.env)
        """
        self.config_path = config_path
        self.prompt_path = prompt_path
        self.alert_config_path = alert_config_path
        
        # Carrega configurações
        self.config = load_yaml(config_path)
        self.prompts = read_prompts(prompt_path)
        
        # Adiciona diretório do config para resolução de caminhos relativos
        config_dir = Path(config_path).parent
        self.config["_config_dir"] = str(config_dir)
        self.config["alert_config_path"] = alert_config_path
        
        # Logger
        log_file = self.config.get("log_file", "pipeline.log")
        log_level = self.config.get("log_level", "INFO")
        level_map = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50
        }
        logger = CentralizedLogger(log_file=log_file, level=level_map.get(log_level.upper(), 20))
        set_logger(logger)
        self.logger = get_logger()
        
        # Modelos compartilhados (serão carregados)
        self.person_detector = None
        self.ppe_detector = None
        
        # Threads de câmeras
        self.camera_threads: Dict[str, CameraThread] = {}
        
        # Buffer compartilhado para streaming (thread-safe)
        self.shared_frame_buffer: Dict[str, bytes] = {}
        self.frame_buffer_lock = threading.Lock()
        
        # Flag de execução
        self.running = threading.Event()
        self.running.set()  # Inicia como True (não parado)
        
        # Servidor de streaming
        self.stream_server: Optional[StreamServer] = None
        self.stream_server_thread: Optional[threading.Thread] = None
        
        # Registra handlers de sinal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handler para sinais de interrupção."""
        self.logger.info("MAIN", f"Recebido sinal {signum}. Iniciando shutdown graceful...")
        self.stop()
    
    def load_models(self):
        """Carrega modelos de detecção (compartilhados entre threads)."""
        self.logger.info("MAIN", "Carregando modelos...")
        
        # RT-DETR-X para detecção de pessoas
        try:
            self.person_detector = RTDETRPerson(
                weights=self.config.get("rtdetr_weights", "rtdetr-x.pt"),
                device=self.config.get("device", "cuda"),
                imgsz=self.config.get("rtdetr_imgsz", 1280),
                conf=self.config.get("rtdetr_conf", 0.15),
                iou=self.config.get("rtdetr_iou", 0.45),
                min_area=float(self.config.get("rtdetr_min_area", 0.0001)),
                max_area=float(self.config.get("rtdetr_max_area", 0.8)),
                min_aspect_ratio=float(self.config.get("rtdetr_min_aspect_ratio", 0.25)),
                max_aspect_ratio=float(self.config.get("rtdetr_max_aspect_ratio", 6.0)),
                min_height_px=int(self.config.get("rtdetr_min_height_px", 20)),
                min_width_px=int(self.config.get("rtdetr_min_width_px", 10)),
                disable_filters=bool(self.config.get("rtdetr_disable_filters", False)),
                debug=self.config.get("debug", False)
            )
            self.logger.info("MAIN", "RT-DETR-X carregado com sucesso")
        except Exception as e:
            self.logger.error("MAIN", f"Falha ao carregar RT-DETR-X: {e}")
            self.logger.exception("MAIN", f"Traceback: {traceback.format_exc()}")
            raise
        
        # Detector de EPIs unificado
        ppe_detector_type = self.config.get("ppe_detector", "yolo-world").lower()
        detector_kwargs = {
            "device": self.config.get("device", "cuda"),
        }
        
        if ppe_detector_type == "yolo-world":
            detector_kwargs.update({
                "yw_model": self.config.get("yw_model", "yolov8m-world.pt"),
                "yw_fp16": bool(self.config.get("yw_fp16", True)),
                "yw_use_crop": bool(self.config.get("yw_use_crop", False)),
                "yw_crop_padding": float(self.config.get("yw_crop_padding", 0.20)),
                "yw_min_crop_size": int(self.config.get("yw_min_crop_size", 32)),
                "yw_imgsz": int(self.config.get("yw_imgsz", 1280)),
            })
        elif ppe_detector_type == "owl-v2":
            detector_kwargs.update({
                "ovd_model": self.config.get("ovd_model", "google/owlv2-base-patch16"),
                "ovd_fp16": bool(self.config.get("ovd_fp16", True)),
                "ovd_cache_dir": self.config.get("ovd_cache_dir", "./.hf"),
                "ovd_use_fast": bool(self.config.get("ovd_use_fast", True)),
                "ovd_quantization_mode": self.config.get("ovd_quantization_mode", "none"),
            })
        else:
            raise ValueError(f"Tipo de detector PPE não suportado: {ppe_detector_type}")
        
        try:
            self.ppe_detector = UnifiedPPEDetector(detector_type=ppe_detector_type, **detector_kwargs)
            self.logger.info("MAIN", f"Detector de EPIs ({ppe_detector_type.upper()}) carregado com sucesso")
        except Exception as e:
            self.logger.error("MAIN", f"Falha ao carregar detector de EPIs: {e}")
            self.logger.exception("MAIN", f"Traceback: {traceback.format_exc()}")
            raise
    
    def start_camera_threads(self):
        """Inicia threads para cada câmera configurada."""
        cameras = self.config.get("cameras", [])
        if not cameras:
            self.logger.warning("MAIN", "Nenhuma câmera configurada")
            return
        
        self.logger.info("MAIN", f"Iniciando {len(cameras)} thread(s) de câmera(s)...")
        
        for cam_config in cameras:
            camera_id = cam_config.get("id")
            camera_uri = cam_config.get("uri")
            enabled = cam_config.get("enabled", True)
            
            if not camera_id or not camera_uri:
                self.logger.warning("MAIN", f"Câmera sem ID ou URI: {cam_config}")
                continue
            
            if not enabled:
                self.logger.info("MAIN", f"Câmera {camera_id} desabilitada. Pulando...")
                continue
            
            try:
                thread = CameraThread(
                    camera_id=camera_id,
                    camera_uri=camera_uri,
                    camera_config=cam_config,
                    global_config=self.config,
                    prompts=self.prompts,
                    person_detector=self.person_detector,
                    ppe_detector=self.ppe_detector,
                    shared_frame_buffer=self.shared_frame_buffer,
                    frame_buffer_lock=self.frame_buffer_lock,
                    running_flag=self.running
                )
                thread.start()
                self.camera_threads[camera_id] = thread
                self.logger.info("MAIN", f"Thread de câmera {camera_id} iniciada (PID: {thread.ident})")
            except FileNotFoundError as e:
                # Erro de arquivo ROI não encontrado - loga mas continua com outras câmeras
                self.logger.error("MAIN", f"Erro ao iniciar thread de câmera {camera_id}: {e}")
                self.logger.warning("MAIN", f"Câmera {camera_id} será pulada. Continuando com outras câmeras...")
            except Exception as e:
                # Outros erros - loga mas continua com outras câmeras
                self.logger.error("MAIN", f"Erro ao iniciar thread de câmera {camera_id}: {e}")
                self.logger.exception("MAIN", f"Traceback: {traceback.format_exc()}")
                self.logger.warning("MAIN", f"Câmera {camera_id} será pulada. Continuando com outras câmeras...")
    
    def start_stream_server(self):
        """Inicia servidor de streaming MJPEG."""
        stream_enabled = self.config.get("stream_enabled", True)
        if not stream_enabled:
            self.logger.info("MAIN", "Streaming desabilitado no YAML")
            return
        
        stream_port = int(self.config.get("stream_port", 5000))
        stream_fps = int(self.config.get("stream_fps", 10))
        stream_jpeg_quality = int(self.config.get("stream_jpeg_quality", 85))
        
        try:
            # Cria dicionário de locks por câmera (para compatibilidade com StreamServer)
            frame_locks = {cam_id: threading.Lock() for cam_id in self.camera_threads.keys()}
            
            self.stream_server = StreamServer(
                frame_buffer=self.shared_frame_buffer,
                frame_locks=frame_locks,
                port=stream_port,
                fps=stream_fps,
                jpeg_quality=stream_jpeg_quality
            )
            
            self.stream_server_thread = threading.Thread(
                target=self.stream_server.run,
                name="StreamServer",
                daemon=True
            )
            self.stream_server_thread.start()
            self.logger.info("MAIN", f"Servidor de streaming iniciado na porta {stream_port}")
        except Exception as e:
            self.logger.error("MAIN", f"Erro ao iniciar servidor de streaming: {e}")
            self.logger.exception("MAIN", f"Traceback: {traceback.format_exc()}")
    
    def run(self):
        """Executa o pipeline principal."""
        try:
            # Carrega modelos
            self.load_models()
            
            # Inicia threads de câmeras
            self.start_camera_threads()
            
            if not self.camera_threads:
                self.logger.error("MAIN", "Nenhuma thread de câmera iniciada. Encerrando.")
                return
            
            # Inicia servidor de streaming
            self.start_stream_server()
            
            self.logger.info("MAIN", "Pipeline iniciado. Aguardando threads...")
            self.logger.info("MAIN", f"Câmeras ativas: {list(self.camera_threads.keys())}")
            
            # Loop de monitoramento
            while self.running.is_set():
                # Verifica se threads ainda estão vivas
                dead_threads = []
                for cam_id, thread in self.camera_threads.items():
                    if not thread.is_alive():
                        dead_threads.append(cam_id)
                        self.logger.warning("MAIN", f"Thread de câmera {cam_id} morreu. Não será reiniciada automaticamente.")
                
                # Remove threads mortas
                for cam_id in dead_threads:
                    del self.camera_threads[cam_id]
                
                if not self.camera_threads:
                    self.logger.warning("MAIN", "Todas as threads de câmera morreram. Encerrando.")
                    break
                
                time.sleep(5.0)  # Verifica a cada 5 segundos
        
        except KeyboardInterrupt:
            self.logger.info("MAIN", "Interrompido pelo usuário")
        except Exception as e:
            self.logger.critical("MAIN", f"Erro fatal: {e}")
            self.logger.exception("MAIN", f"Traceback: {traceback.format_exc()}")
        finally:
            self.stop()
    
    def stop(self):
        """Para o pipeline gracefulmente."""
        self.logger.info("MAIN", "Parando pipeline...")
        
        # Sinaliza para threads pararem
        self.running.clear()
        
        # Aguarda threads de câmeras terminarem
        for cam_id, thread in list(self.camera_threads.items()):
            self.logger.info("MAIN", f"Aguardando thread de câmera {cam_id} terminar...")
            thread.join(timeout=10.0)
            if thread.is_alive():
                self.logger.warning("MAIN", f"Thread de câmera {cam_id} não terminou em 10s")
        
        # Para servidor de streaming
        if self.stream_server:
            self.stream_server.stop()
            if self.stream_server_thread and self.stream_server_thread.is_alive():
                self.stream_server_thread.join(timeout=5.0)
        
        self.logger.info("MAIN", "Pipeline finalizado")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Pipeline multi-câmera RT-DETR-X + PPE Detection")
    parser.add_argument("--config", default="config/stream_rtdetr_cam.yaml", help="Caminho do arquivo YAML de configuração")
    parser.add_argument("--prompts", default="config/ppe_prompts_rtdetr.yaml", help="Caminho do arquivo YAML de prompts")
    parser.add_argument("--alert-config", default="db_config.env", help="Caminho do arquivo de configuração de alertas")
    
    args = parser.parse_args()
    
    # Verifica se arquivos existem
    if not os.path.exists(args.config):
        print(f"[ERROR] Arquivo de configuração não encontrado: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.prompts):
        print(f"[ERROR] Arquivo de prompts não encontrado: {args.prompts}")
        sys.exit(1)
    
    # Cria e executa pipeline
    manager = PipelineManager(
        config_path=args.config,
        prompt_path=args.prompts,
        alert_config_path=args.alert_config
    )
    
    manager.run()


if __name__ == "__main__":
    main()

