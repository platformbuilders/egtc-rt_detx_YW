"""
Stream Server - Servidor HTTP para streaming MJPEG das câmeras.
Serve frames processados em tempo real via HTTP multipart/x-mixed-replace.
"""
import time
import threading
import cv2
import numpy as np
import pickle
from typing import Dict, Optional
from flask import Flask, Response
from logger import get_logger


class StreamServer:
    """
    Servidor HTTP para streaming MJPEG das câmeras.
    """
    
    def __init__(self, frame_buffer: Dict[str, bytes], 
                 frame_locks: Dict[str, threading.Lock],
                 port: int = 5000,
                 fps: int = 10,
                 jpeg_quality: int = 85):
        """
        Inicializa o servidor de streaming.
        
        Args:
            frame_buffer: Dicionário compartilhado {camera_id: frame_bytes} com frames pickle
            frame_locks: Dicionário de locks por câmera para acesso thread-safe
            port: Porta do servidor HTTP
            fps: FPS do stream (frames por segundo)
            jpeg_quality: Qualidade JPEG (1-100, maior = melhor qualidade)
        """
        self.frame_buffer = frame_buffer
        self.frame_locks = frame_locks
        self.port = port
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.logger = get_logger()
        
        # Cria app Flask
        self.app = Flask(__name__)
        self.app.add_url_rule('/stream/<camera_id>', 'stream_camera', self._stream_camera, methods=['GET'])
        self.app.add_url_rule('/health', 'health', self._health, methods=['GET'])
        self.app.add_url_rule('/cameras', 'list_cameras', self._list_cameras, methods=['GET'])
        self.app.add_url_rule('/test', 'test', self._test, methods=['GET'])
        
        # Desabilita logging do Flask (já temos nosso logger)
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        # Thread do servidor
        self.server_thread: Optional[threading.Thread] = None
        self.running = True
    
    def _test(self):
        """Endpoint de teste - retorna uma imagem estática."""
        import io
        # Cria imagem de teste
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Stream Server OK", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        success, jpeg_buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            return Response(jpeg_buffer.tobytes(), mimetype='image/jpeg')
        return "Error", 500
    
    def _health(self):
        """Endpoint de health check."""
        return {"status": "ok", "cameras": list(self.frame_buffer.keys())}
    
    def _list_cameras(self):
        """Lista câmeras disponíveis."""
        return {"cameras": list(self.frame_buffer.keys())}
    
    def _stream_camera(self, camera_id: str):
        """
        Endpoint para streaming MJPEG de uma câmera.
        
        Args:
            camera_id: ID da câmera (ex: "CAM063")
        
        Returns:
            Response HTTP com stream MJPEG
        """
        self.logger.info("STREAM_SERVER", f"Cliente conectado ao stream de {camera_id}")
        
        def generate():
            """Generator para frames MJPEG."""
            self.logger.info("STREAM_SERVER", f"Generator iniciado para {camera_id}")
            frame_interval = 1.0 / self.fps
            last_frame_time = 0
            frames_sent = 0
            no_frame_count = 0
            
            # Envia primeiro frame imediatamente (placeholder se necessário)
            try:
                frame = None
                if camera_id in self.frame_buffer:
                    # Usa lock se disponível
                    if camera_id in self.frame_locks:
                        with self.frame_locks[camera_id]:
                            frame_bytes = self.frame_buffer.get(camera_id)
                    else:
                        frame_bytes = self.frame_buffer.get(camera_id)
                    
                    if frame_bytes:
                        try:
                            frame = pickle.loads(frame_bytes)
                        except Exception as e:
                            self.logger.error("STREAM_SERVER", f"Erro ao deserializar frame: {e}")
                            frame = None
                
                if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    # Cria placeholder
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, f"Aguardando {camera_id}...", (50, 240), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Envia primeiro frame
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                success, jpeg_buffer = cv2.imencode('.jpg', frame, encode_params)
                if success:
                    self.logger.info("STREAM_SERVER", f"Enviando primeiro frame para {camera_id}")
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + 
                           jpeg_buffer.tobytes() + b'\r\n')
                    frames_sent += 1
                    last_frame_time = time.time()
            except Exception as e:
                self.logger.error("STREAM_SERVER", f"Erro ao enviar primeiro frame: {e}")
                import traceback
                self.logger.exception("STREAM_SERVER", f"Traceback: {traceback.format_exc()}")
            
            while self.running:
                try:
                    # Verifica se câmera existe no buffer
                    buffer_keys = list(self.frame_buffer.keys()) if hasattr(self.frame_buffer, 'keys') else []
                    if camera_id not in self.frame_buffer:
                        no_frame_count += 1
                        if no_frame_count == 1:
                            self.logger.warning("STREAM_SERVER", f"Câmera {camera_id} não encontrada no buffer. Câmeras disponíveis: {buffer_keys}. Aguardando frames...")
                        elif no_frame_count % 10 == 0:  # Log a cada 10 tentativas
                            self.logger.warning("STREAM_SERVER", f"Ainda aguardando {camera_id}... Câmeras no buffer: {buffer_keys}")
                        # Envia frame placeholder enquanto aguarda
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(placeholder, f"Aguardando {camera_id}...", (50, 240), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(placeholder, f"Buffer: {len(buffer_keys)} cams", (50, 280), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                        success, jpeg_buffer = cv2.imencode('.jpg', placeholder, encode_params)
                        if success:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + 
                                   jpeg_buffer.tobytes() + b'\r\n')
                        time.sleep(1.0)
                        continue
                    
                    no_frame_count = 0
                    
                    # Pega frame do buffer (thread-safe)
                    frame = None
                    try:
                        if camera_id in self.frame_buffer:
                            # Usa lock se disponível
                            if camera_id in self.frame_locks:
                                with self.frame_locks[camera_id]:
                                    frame_bytes = self.frame_buffer.get(camera_id)
                            else:
                                frame_bytes = self.frame_buffer.get(camera_id)
                            
                            if frame_bytes:
                                try:
                                    frame = pickle.loads(frame_bytes)
                                except Exception as e:
                                    if frames_sent % 30 == 0:
                                        self.logger.error("STREAM_SERVER", f"Erro ao deserializar frame de {camera_id}: {e}")
                                    frame = None
                    except Exception as e:
                        if frames_sent % 30 == 0:
                            self.logger.error("STREAM_SERVER", f"Erro ao acessar buffer de {camera_id}: {e}")
                        frame = None
                    
                    if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                        # Converte BGR para RGB (OpenCV usa BGR, mas JPEG é RGB)
                        # Na verdade, cv2.imencode funciona com BGR, então mantemos BGR
                        
                        # Codifica frame como JPEG
                        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                        success, jpeg_buffer = cv2.imencode('.jpg', frame, encode_params)
                        
                        if success:
                            # Envia frame via HTTP multipart
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + 
                                   jpeg_buffer.tobytes() + b'\r\n')
                            
                            frames_sent += 1
                            if frames_sent == 1:
                                self.logger.info("STREAM_SERVER", f"Primeiro frame enviado para {camera_id}")
                            
                            # Controla FPS
                            current_time = time.time()
                            elapsed = current_time - last_frame_time
                            if elapsed < frame_interval:
                                time.sleep(frame_interval - elapsed)
                            last_frame_time = time.time()
                        else:
                            self.logger.warning("STREAM_SERVER", f"Falha ao codificar frame JPEG para {camera_id}")
                            time.sleep(0.1)
                    else:
                        # Frame não disponível ou inválido, aguarda
                        time.sleep(0.1)
                
                except Exception as e:
                    self.logger.error("STREAM_SERVER", f"Erro no stream de {camera_id}: {e}")
                    time.sleep(0.5)
        
        return Response(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    
    def run(self):
        """Inicia o servidor (compatibilidade com threading.Thread)."""
        if self.server_thread and self.server_thread.is_alive():
            return
        
        self.running = True
        
        def run_server():
            try:
                self.logger.info("STREAM_SERVER", f"Iniciando servidor MJPEG na porta {self.port}")
                self.app.run(host='0.0.0.0', port=self.port, threaded=True, debug=False, use_reloader=False)
            except Exception as e:
                self.logger.error("STREAM_SERVER", f"Erro no servidor HTTP: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True, name="StreamServer")
        self.server_thread.start()
        self.logger.info("STREAM_SERVER", f"Servidor MJPEG iniciado. Endpoints: /stream/<camera_id>, /health, /cameras")
    
    def stop(self):
        """Para o servidor HTTP."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
        self.logger.info("STREAM_SERVER", "Servidor MJPEG parado")

