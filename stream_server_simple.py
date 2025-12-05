"""
Servidor MJPEG simples - Serve frames salvos pelo pipeline.
Cada pipeline salva frames em out/stream/{CAM_ID}.jpg
"""
import os
import time
import threading
from pathlib import Path
from flask import Flask, Response
import cv2
import numpy as np


class SimpleStreamServer:
    """Servidor MJPEG que serve frames salvos em disco."""
    
    def __init__(self, stream_dir: str = "./out/stream", port: int = 5000, fps: int = 10):
        self.stream_dir = Path(stream_dir)
        self.port = port
        self.fps = fps
        self.app = Flask(__name__)
        self.running = False
        self.server_thread = None
        
        # Endpoints
        self.app.add_url_rule('/stream/<camera_id>', 'stream_camera', self._stream_camera, methods=['GET'])
        self.app.add_url_rule('/health', 'health', self._health, methods=['GET'])
        self.app.add_url_rule('/cameras', 'list_cameras', self._list_cameras, methods=['GET'])
        
        # Desabilita logging do Flask
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
    
    def _health(self):
        """Health check."""
        return {"status": "ok", "stream_dir": str(self.stream_dir)}
    
    def _list_cameras(self):
        """Lista câmeras disponíveis."""
        cameras = []
        if self.stream_dir.exists():
            for f in self.stream_dir.glob("*.jpg"):
                cameras.append(f.stem)
        return {"cameras": cameras}
    
    def _stream_camera(self, camera_id: str):
        """Endpoint de streaming MJPEG."""
        def generate():
            frame_path = self.stream_dir / f"{camera_id}.jpg"
            frame_interval = 1.0 / self.fps
            last_frame_time = 0
            
            while True:
                try:
                    if frame_path.exists():
                        # Lê frame
                        frame = cv2.imread(str(frame_path))
                        if frame is not None:
                            # Codifica como JPEG
                            success, jpeg_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if success:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + 
                                       jpeg_buffer.tobytes() + b'\r\n')
                                
                                # Controla FPS
                                current_time = time.time()
                                elapsed = current_time - last_frame_time
                                if elapsed < frame_interval:
                                    time.sleep(frame_interval - elapsed)
                                last_frame_time = time.time()
                    else:
                        # Frame não existe - envia placeholder
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(placeholder, f"Aguardando {camera_id}...", (50, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        success, jpeg_buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if success:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + 
                                   jpeg_buffer.tobytes() + b'\r\n')
                        time.sleep(1.0)
                
                except Exception as e:
                    print(f"[ERROR] Erro no stream de {camera_id}: {e}")
                    time.sleep(0.5)
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def run(self):
        """Inicia servidor."""
        if self.server_thread and self.server_thread.is_alive():
            return
        
        self.running = True
        
        def run_server():
            print(f"[INFO] Servidor MJPEG iniciado na porta {self.port}")
            print(f"[INFO] Endpoints: /stream/<camera_id>, /health, /cameras")
            self.app.run(host='0.0.0.0', port=self.port, threaded=True, debug=False, use_reloader=False)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
    
    def stop(self):
        """Para servidor."""
        self.running = False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Servidor MJPEG simples")
    parser.add_argument("--stream-dir", default="./out/stream", help="Diretório com frames")
    parser.add_argument("--port", type=int, default=5000, help="Porta HTTP")
    parser.add_argument("--fps", type=int, default=10, help="FPS do stream")
    
    args = parser.parse_args()
    
    server = SimpleStreamServer(stream_dir=args.stream_dir, port=args.port, fps=args.fps)
    
    try:
        server.run()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Parando servidor...")
        server.stop()

