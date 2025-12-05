########################################################################################
#                                                                                      #
#                                                                                      #
#    ██████╗░██╗░░░██╗██╗██╗░░░░░██████╗░███████╗██████╗░░██████╗                      #
#    ██╔══██╗██║░░░██║██║██║░░░░░██╔══██╗██╔════╝██╔══██╗██╔════╝░                     #
#    ██████╦╝██║░░░██║██║██║░░░░░██║░░██║█████╗░░██████╔╝╚█████╗░░                     #
#    ██╔══██╗██║░░░██║██║██║░░░░░██║░░██║██╔══╝░░██╔══██╗░╚═══██╗                      #
#    ██████╦╝╚██████╔╝██║███████╗██████╔╝███████╗██║░░██║██████╔╝                      #
#    ╚═════╝░░╚═════╝░╚═╝╚══════╝╚═════╝░╚══════╝╚═╝░░╚═╝╚═════╝░                      #
#                                                                                      #
#    → Builders – Construímos o futuro, linha por linha.                               #
#    → https://paltformbuilders.io  |  contato@platformbuilders.io                     #
#                                                                                      #
########################################################################################


# Sistema Supervisor Multiprocessing

## Visão Geral

Sistema de orquestração multiprocessing para processar múltiplas câmeras simultaneamente, com:
- **Isolamento total**: Cada câmera roda em processo separado
- **GPU compartilhada**: Modelos carregados uma vez, processamento sequencial
- **Telemetria**: Monitora processos, reinicia automaticamente
- **Logger centralizado**: Logs unificados com identificação de câmera
- **Fila assíncrona**: Performance otimizada com processamento não-bloqueante

## Arquitetura

```
Supervisor (Processo Principal)
├── GPU Worker (1 processo)
│   ├── Carrega RT-DETR-X (uma vez)
│   ├── Carrega PPE Detector (uma vez)
│   └── Processa frames sequencialmente
│
└── Camera Workers (N processos, 1 por câmera)
    ├── Camera Worker 1
    ├── Camera Worker 2
    ├── Camera Worker 3
    └── Camera Worker 4
```

### Fluxo de Processamento

1. **Camera Worker** captura frame da câmera
2. **Camera Worker** serializa frame e envia para fila GPU (assíncrono)
3. **GPU Worker** recebe frame, processa (RT-DETR + PPE), retorna resultado
4. **Camera Worker** recebe resultado, faz tracking, alertas, desenho
5. **Camera Worker** salva vídeo/exibe frame

## Componentes

### 1. `supervisor.py`
Processo principal que:
- Inicializa GPU Worker e Camera Workers
- Monitora health dos processos (heartbeat)
- Reinicia processos que crasharem (com limite configurável)
- Gerencia shutdown graceful

### 2. `gpu_worker.py`
Worker dedicado para processamento GPU:
- Carrega modelos uma vez (RT-DETR-X + PPE Detector)
- Processa frames sequencialmente
- Retorna resultados via filas de resposta

### 3. `camera_worker.py`
Worker para cada câmera:
- Captura frames da câmera
- Envia frames para GPU Worker (assíncrono)
- Processa resultados (tracking, alertas, desenho)
- Salva vídeo e métricas

### 4. `logger.py`
Logger centralizado thread-safe:
- Logs unificados em arquivo único
- Identificação de câmera em cada mensagem
- Thread-safe para escrita concorrente

## Uso

### Linha de Comando

```bash
python3 supervisor.py \
    --config config/stream_rtdetr_cam63.yaml \
    --prompts config/ppe_prompts_rtdetr.yaml \
    --roi rois/roi_cam61.json \
    --roi-polys roi_epi_on \
    --log-file pipeline.log \
    --heartbeat-timeout 30.0 \
    --max-restarts 5 \
    --restart-window 3600.0
```

### Parâmetros

- `--config`: Caminho do arquivo YAML de configuração (obrigatório)
- `--prompts`: Caminho do arquivo YAML de prompts (obrigatório)
- `--roi`: Caminho do arquivo ROI (opcional)
- `--roi-polys`: Lista de polígonos ROI a usar (opcional)
- `--log-file`: Caminho do arquivo de log (default: `pipeline.log`)
- `--heartbeat-timeout`: Timeout de heartbeat em segundos (default: 30.0)
- `--max-restarts`: Máximo de reinicializações por processo (default: 5)
- `--restart-window`: Janela de tempo para contar reinicializações em segundos (default: 3600.0 = 1h)

### Configuração YAML

Adicione as seguintes configurações no YAML (opcional, valores padrão serão usados se não especificado):

```yaml
# ==== Supervisor (Multiprocessing) ====
supervisor_heartbeat_timeout: 30.0  # Timeout em segundos
supervisor_max_restarts: 5          # Máximo de reinicializações
supervisor_restart_window: 3600.0   # Janela de tempo (segundos)
```

## Monitoramento

### Heartbeat

Cada Camera Worker envia heartbeat a cada 5 segundos. Se o Supervisor não receber heartbeat por `heartbeat_timeout` segundos, considera o processo morto e reinicia.

### Reinicialização

- Limite de `max_restarts` reinicializações por processo em uma janela de `restart_window` segundos
- Se o limite for atingido, o processo não será mais reiniciado (evita loops infinitos)
- Logs detalhados de cada reinicialização

### Logs

Logs centralizados em arquivo único com formato:
```
[TIMESTAMP] [CAMERA_ID] [LEVEL] Mensagem
```

Exemplo:
```
2024-01-15 10:30:45 [CAM063] [INFO] Camera Worker iniciado
2024-01-15 10:30:46 [CAM063] [ERROR] Erro no processamento GPU: ...
2024-01-15 10:30:47 [SUPERVISOR] [WARNING] CAM063: Heartbeat timeout (30s)
```

## Vantagens

1. **Isolamento**: Processo crashado não afeta outros
2. **Economia de VRAM**: Modelos carregados uma vez (~4GB por GPU Worker)
3. **Telemetria**: Logs individuais por câmera
4. **Robustez**: Reinicialização automática
5. **Performance**: Fila assíncrona, processamento não-bloqueante
6. **Escalabilidade**: Fácil adicionar mais câmeras

## Limitações

1. **Serialização**: Frames são serializados (pickle) para passar entre processos (overhead mínimo)
2. **GPU Sequencial**: Processamento GPU é sequencial (um frame por vez)
3. **VRAM**: GPU Worker precisa de VRAM suficiente para modelos (~2GB para RT-DETR-L + YOLO-World-M)

## Troubleshooting

### Processo não reinicia
- Verifique se `max_restarts` não foi atingido
- Verifique logs para erros recorrentes
- Aumente `restart_window` se necessário

### Fila GPU cheia
- Reduza `target_fps` no YAML
- Aumente tamanho das filas (código)
- Verifique se GPU Worker está processando corretamente

### Heartbeat timeout
- Verifique se câmera está funcionando
- Aumente `heartbeat_timeout` se necessário
- Verifique logs do Camera Worker específico

## Comparação com Pipeline Single-Process


