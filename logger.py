"""
Logger centralizado thread-safe para sistema multiprocessing.
Suporta identificação de câmera e escrita concorrente em arquivo único.
"""
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


class CentralizedLogger:
    """
    Logger centralizado que suporta múltiplos processos/threads.
    Cada mensagem inclui identificação de câmera e timestamp.
    """
    
    def __init__(self, log_file: str = "pipeline.log", level: int = logging.INFO):
        """
        Inicializa o logger centralizado.
        
        Args:
            log_file: Caminho do arquivo de log (único para todas as câmeras)
            level: Nível de log (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_file = Path(log_file)
        self.level = level
        self._lock = threading.Lock()  # Lock para escrita thread-safe
        
        # Cria diretório se não existir
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configura logger Python padrão
        self.logger = logging.getLogger("pipeline")
        self.logger.setLevel(level)
        
        # Remove handlers existentes
        self.logger.handlers.clear()
        
        # Handler para arquivo (thread-safe)
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Handler para console (opcional)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _log(self, level: int, camera_id: str, message: str):
        """
        Método interno thread-safe para logging.
        
        Args:
            level: Nível de log (logging.DEBUG, INFO, etc.)
            camera_id: ID da câmera (ex: "CAM063")
            message: Mensagem a ser logada
        """
        with self._lock:
            # Adiciona identificação de câmera à mensagem
            formatted_message = f"[{camera_id}] {message}"
            self.logger.log(level, formatted_message)
    
    def debug(self, camera_id: str, message: str):
        """Log de debug."""
        self._log(logging.DEBUG, camera_id, message)
    
    def info(self, camera_id: str, message: str):
        """Log de informação."""
        self._log(logging.INFO, camera_id, message)
    
    def warning(self, camera_id: str, message: str):
        """Log de aviso."""
        self._log(logging.WARNING, camera_id, message)
    
    def error(self, camera_id: str, message: str):
        """Log de erro."""
        self._log(logging.ERROR, camera_id, message)
    
    def critical(self, camera_id: str, message: str):
        """Log crítico."""
        self._log(logging.CRITICAL, camera_id, message)
    
    def exception(self, camera_id: str, message: str):
        """Log de exceção (inclui traceback)."""
        with self._lock:
            formatted_message = f"[{camera_id}] {message}"
            self.logger.exception(formatted_message)


# Instância global do logger (será inicializada pelo supervisor)
_global_logger: Optional[CentralizedLogger] = None
_logger_lock = threading.Lock()


def get_logger() -> CentralizedLogger:
    """
    Retorna a instância global do logger.
    Se não existir, cria uma instância padrão.
    """
    global _global_logger
    if _global_logger is None:
        with _logger_lock:
            if _global_logger is None:
                _global_logger = CentralizedLogger()
    return _global_logger


def set_logger(logger: CentralizedLogger):
    """Define a instância global do logger."""
    global _global_logger
    with _logger_lock:
        _global_logger = logger

