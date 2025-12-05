"""
Scheduler - Executa pipeline original para cada câmera em processo separado.
Solução simples e confiável baseada no pipeline que já funciona.
"""
import os
import sys
import time
import signal
import subprocess
import argparse
import yaml
from pathlib import Path
from typing import List, Dict


class CameraScheduler:
    """Gerencia múltiplas instâncias do pipeline, uma por câmera."""
    
    def __init__(self, config_path: str, prompt_path: str, alert_config_path: str = "db_config.env"):
        self.config_path = config_path
        self.prompt_path = prompt_path
        self.alert_config_path = alert_config_path
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = True
        
        # Carrega configuração
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Registra handlers de sinal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handler para sinais de interrupção."""
        print(f"\n[INFO] Recebido sinal {signum}. Parando todas as câmeras...")
        self.stop()
        sys.exit(0)
    
    def _create_camera_config(self, camera_id: str, camera_config: Dict) -> str:
        """Cria um YAML temporário para uma única câmera."""
        temp_dir = Path("/tmp/egtc_detr")
        temp_dir.mkdir(exist_ok=True)
        
        # Cria config para uma única câmera
        single_cam_config = self.config.copy()
        single_cam_config["cameras"] = [camera_config]
        
        # IMPORTANTE: Move roi_ppe_config da câmera para o nível global
        # O pipeline lê roi_ppe_config do nível global, não de cameras
        if "roi_ppe_config" in camera_config:
            single_cam_config["roi_ppe_config"] = camera_config["roi_ppe_config"]
        elif "global_roi_ppe_config" in self.config:
            single_cam_config["roi_ppe_config"] = self.config["global_roi_ppe_config"]
        
        # Salva em arquivo temporário
        temp_config_path = temp_dir / f"config_{camera_id}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(single_cam_config, f, default_flow_style=False)
        
        return str(temp_config_path)
    
    def start_camera(self, camera_id: str, camera_config: Dict):
        """Inicia pipeline para uma câmera em processo separado."""
        if camera_id in self.processes:
            proc = self.processes[camera_id]
            if proc.poll() is None:  # Processo ainda está rodando
                print(f"[WARN] {camera_id}: Processo já está rodando")
                return
        
        enabled = camera_config.get("enabled", True)
        if not enabled:
            print(f"[INFO] {camera_id}: Câmera desabilitada. Pulando...")
            return
        
        # Cria config temporário para esta câmera
        temp_config = self._create_camera_config(camera_id, camera_config)
        
        # Comando para executar o pipeline
        script_dir = Path(__file__).parent
        pipeline_script = script_dir / "pipeline_RETDETRX_YW.py"
        
        cmd = [
            sys.executable,
            str(pipeline_script),
            "--config", temp_config,
            "--prompts", self.prompt_path,
            "--alert-config", self.alert_config_path
        ]
        
        # IMPORTANTE: Passa roi_path e roi_polys via CLI (o pipeline espera esses parâmetros)
        roi_path = camera_config.get("roi_path")
        if roi_path:
            # Resolve caminho relativo ao diretório do projeto
            script_dir = Path(__file__).parent
            if not Path(roi_path).is_absolute():
                roi_path = str(script_dir / roi_path)
            cmd.extend(["--roi", roi_path])
        
        roi_polys = camera_config.get("roi_polys", [])
        if roi_polys:
            cmd.extend(["--roi-polys", ",".join(roi_polys)])
        
        # Adiciona flags opcionais do YAML
        # Sempre desenha ROI se houver ROI configurado (para aparecer no streaming)
        if roi_path or self.config.get("draw_roi", False):
            cmd.append("--draw-roi")
        if self.config.get("show_video", False):
            cmd.append("--show-video")
        if self.config.get("enable_alerts", False):
            cmd.append("--enable-alerts")
        if self.config.get("show_alert_grid", False):
            cmd.append("--show-alert-grid")
        if self.config.get("show_rtdetr_boxes", False):
            cmd.append("--show-rtdetr-boxes")
        if self.config.get("debug", False):
            cmd.append("--debug")
        
        print(f"[INFO] Iniciando {camera_id}...")
        print(f"[DEBUG] Comando: {' '.join(cmd)}")
        
        try:
            # Inicia processo
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes[camera_id] = proc
            
                # Thread para ler output (opcional, para logs)
            import threading
            def read_output():
                try:
                    for line in proc.stdout:
                        if line:
                            print(f"[{camera_id}] {line.rstrip()}")
                except Exception:
                    pass  # Processo pode ter terminado
            
            thread = threading.Thread(target=read_output, daemon=True)
            thread.start()
            
            print(f"[INFO] {camera_id}: Processo iniciado (PID: {proc.pid})")
            
        except Exception as e:
            print(f"[ERROR] {camera_id}: Erro ao iniciar processo: {e}")
            import traceback
            traceback.print_exc()
    
    def start_all(self):
        """Inicia pipeline para todas as câmeras configuradas."""
        cameras = self.config.get("cameras", [])
        if not cameras:
            print("[ERROR] Nenhuma câmera configurada")
            return
        
        print(f"[INFO] Iniciando {len(cameras)} câmera(s)...")
        
        # Inicia servidor de streaming se habilitado
        stream_enabled = self.config.get("stream_enabled", True)
        if stream_enabled:
            self._start_stream_server()
        
        for cam_config in cameras:
            camera_id = cam_config.get("id")
            if not camera_id:
                print(f"[WARN] Câmera sem ID: {cam_config}")
                continue
            
            self.start_camera(camera_id, cam_config)
            time.sleep(1)  # Pequeno delay entre inícios
        
        print(f"[INFO] {len(self.processes)} processo(s) iniciado(s)")
    
    def _start_stream_server(self):
        """Inicia servidor de streaming em thread separada."""
        try:
            from stream_server_simple import SimpleStreamServer
            
            stream_dir = self.config.get("out_dir", "./out")
            stream_dir = os.path.join(stream_dir, "stream")
            stream_port = int(self.config.get("stream_port", 5000))
            stream_fps = int(self.config.get("stream_fps", 10))
            
            server = SimpleStreamServer(stream_dir=stream_dir, port=stream_port, fps=stream_fps)
            server.run()
            
            print(f"[INFO] Servidor MJPEG iniciado na porta {stream_port}")
            print(f"[INFO] Acesse: http://<ip>:{stream_port}/stream/<CAMERA_ID>")
        except Exception as e:
            print(f"[WARN] Não foi possível iniciar servidor de streaming: {e}")
    
    def monitor(self):
        """Monitora processos e reinicia se necessário."""
        while self.running:
            time.sleep(5)  # Verifica a cada 5 segundos
            
            for camera_id, proc in list(self.processes.items()):
                if proc.poll() is not None:  # Processo terminou
                    exit_code = proc.returncode
                    print(f"[WARN] {camera_id}: Processo terminou com código {exit_code}")
                    # Não reinicia automaticamente - apenas loga
                    del self.processes[camera_id]
            
            if not self.processes:
                print("[WARN] Todos os processos terminaram")
                break
    
    def stop(self):
        """Para todos os processos."""
        print("[INFO] Parando todos os processos...")
        self.running = False
        
        for camera_id, proc in list(self.processes.items()):
            print(f"[INFO] Terminando {camera_id} (PID: {proc.pid})...")
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"[WARN] {camera_id}: Forçando término...")
                proc.kill()
                proc.wait()
            except Exception as e:
                print(f"[ERROR] {camera_id}: Erro ao terminar processo: {e}")
        
        self.processes.clear()
        print("[INFO] Todos os processos parados")


def main():
    parser = argparse.ArgumentParser(description="Scheduler para múltiplas câmeras")
    parser.add_argument("--config", default="config/stream_rtdetr_cam.yaml", help="Config YAML")
    parser.add_argument("--prompts", default="config/ppe_prompts_rtdetr.yaml", help="Prompts YAML")
    parser.add_argument("--alert-config", default="db_config.env", help="Alert config")
    
    args = parser.parse_args()
    
    scheduler = CameraScheduler(
        config_path=args.config,
        prompt_path=args.prompts,
        alert_config_path=args.alert_config
    )
    
    try:
        scheduler.start_all()
        scheduler.monitor()
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuário")
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()

