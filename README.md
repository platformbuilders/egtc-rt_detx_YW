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
#     https://paltformbuilders.io  |  contato@platformbuilders.io                      #
#                                                                                      #
########################################################################################

##Sistema de Detecção de EPI (Equipamentos de Proteção Individual)

Sistema de detecção e monitoramento de EPIs em tempo real utilizando RT-DETR-X para detecção de pessoas e YOLO-World/OWL-V2 para detecção de equipamentos de proteção individual.

## 📋 Características

- **Detecção de Pessoas**: Utiliza RT-DETR-X (Ultralytics) para detecção precisa de pessoas
- **Detecção de EPIs**: Suporta YOLO-World e OWL-V2 para detecção de equipamentos de proteção
- **Rastreamento**: Sistema de rastreamento de pessoas com ByteTrack
- **ROI (Região de Interesse)**: Suporte a polígonos de ROI para áreas específicas
- **Sistema de Alertas**: 
  - Alertas configuráveis com debounce e confirmação
  - Integração com Telegram
  - Persistência em banco de dados (MySQL, PostgreSQL, Oracle)
  - Grid visual de alertas
  - Salvamento automático de imagens (crops ou frames completos)
- **Métricas**: Coleta de métricas de performance e detecção
- **Interface Visual**: Bounding boxes coloridos, painéis informativos e grid de alertas

## 🔧 Requisitos

### Hardware
- GPU NVIDIA com suporte CUDA - Recomendamos modelos T4, T40, L40 ou similares. RTX 30XX e 40XX também são suportadas.
- Mínimo 16GB RAM
- Espaço em disco para vídeos e imagens de alertas - recomendamos uma partição /data separada)

### Software
- Ubuntu Server Minimal versão 22.04
- driver nVidia versão 535 (obrigatório manter esta versão)
- CUDA 12.x (drivers serão instalados automaticamente)
- Python 3.10 ou superior
- **Redis** (obrigatório para sistema de alertas - veja seção dedicada abaixo)
- MySQL/PostgreSQL/Oracle (opcional, para persistência de alertas)


## 📦 Instalação

## 0. Instalação de pré-requisitos de infraestrutura

### 0.1 - Driver nVidia

```bash
sudo apt update
sudo apt install ubuntu-drivers-common
sudo apt install nvidia-driver-535
```

Após a instalação ser concluída, reinicie o sistema

```bash
sudo reboot
```

Teste a instalação do driver com:

```bash
nvidia-smi
```

Você deverá receber uma tela com a versão do driver, o modelo da placa e outras informações

```bash
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.274.02             Driver Version: 535.274.02   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       Off | 00000000:00:04.0 Off |                    0 |
| N/A   68C    P0              31W /  70W |    262MiB / 15360MiB |      1%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|                                                                                       |
|                                                                                       |
+---------------------------------------------------------------------------------------+
```

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd egtc_detr
```

### 2. Crie um ambiente virtual

```bash
python3 -m venv egtc_detr_venv
source egtc_detr_venv/bin/activate  # Linux/Mac
# ou
egtc_detr_venv\Scripts\activate  # Windows
```

### 3. Instale as dependências do sistema (apt-get)

**Opção 1: Usando o script automatizado**
```bash
sudo ./install_system_deps.sh
```

**Opção 2: Instalação manual**
```bash
sudo apt update
sudo apt install -y $(grep -v '^#' apt_requirements.txt | tr '\n' ' ')
```

**Opção 3: Instalação seletiva**
Consulte o arquivo `apt_requirements.txt` e instale apenas os pacotes necessários.

**💡 Nota sobre pacotes opcionais:**

O arquivo `apt_requirements.txt` contém dois tipos de pacotes:
- **Essenciais**: Necessários para o funcionamento básico (Python, Redis, etc.)
- **Opcionais**: Necessários apenas se você precisar compilar bibliotecas Python do código-fonte

Se você instala dependências Python via `pip install -r requirements.txt`, a maioria já vem pré-compilada (wheels) e você **NÃO precisa** dos pacotes opcionais marcados com `#OPCIONAL`.

Os pacotes opcionais são necessários apenas se:
- Você precisar compilar bibliotecas do código-fonte
- Você usar versões específicas não disponíveis como wheels
- Você estiver em uma arquitetura incomum (ARM, etc.)

Para instalar apenas os essenciais:
```bash
sudo apt install -y $(grep -v '^#' apt_requirements.txt | grep -v '^#OPCIONAL' | tr '\n' ' ')
```

Para instalar também os opcionais (se necessário):
```bash
sudo apt install -y $(grep -v '^#' apt_requirements_optional.txt | tr '\n' ' ')
```

**Verificar dependências instaladas:**
```bash
./check_system_deps.sh
```

Este script verifica quais pacotes do `apt_requirements.txt` estão:
- ✅ Instalados e atualizados
- ⚠️ Instalados mas desatualizados
- ❌ Não instalados
- ⚠️ Não encontrados nos repositórios

E fornece comandos sugeridos para instalar/atualizar o que falta.

### 4. Instale as dependências Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**💡 Atualizando requirements.txt:**

Se você instalou novos pacotes no ambiente virtual e quer atualizar o `requirements.txt`:

```bash
./update_requirements.sh
```

Este script:
- Faz backup do `requirements.txt` atual
- Gera um novo `requirements.txt` com todas as dependências instaladas
- Mostra estatísticas e diferenças (opcional)

### 4. Instale e configure o Redis

O Redis é **obrigatório** para o funcionamento do sistema de alertas. Ele é usado para:
- Supressão espacial de alertas (evitar alertas duplicados na mesma região)
- Rastreamento de violações já alertadas (hash de violações)
- Gerenciamento de estado temporário

**Instalação do Redis:**

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install redis
sudo systemctl start redis
sudo systemctl enable redis
```

**macOS:**
```bash
brew install redis
brew services start redis
```

**Windows:**
Baixe e instale do site oficial: https://redis.io/download

**Verificar se o Redis está rodando:**
```bash
redis-cli ping
# Deve retornar: PONG
```

**Configuração do Redis no projeto:**

O Redis é configurado automaticamente através do arquivo `db_config.env`:

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

### 5. Instale dependências opcionais

**Para PostgreSQL:**
```bash
pip install psycopg2-binary
```

**Para Oracle:**
```bash
pip install cx_Oracle
```

**Para OpenAI CLIP (necessário para YOLO-World):**
```bash
pip install openai-clip
```

**Para OWL-V2 (opcional):**
```bash
pip install transformers torch pillow
```

### 6. Baixe os modelos

Os modelos serão baixados automaticamente na primeira execução, ou você pode baixá-los manualmente:

- RT-DETR-X: `rtdetr-x.pt` ou `rtdetr-l.pt`
- YOLO-World: `yolov8m-world.pt` ou `yolov8s-world.pt`

## ⚙️ Configuração

### 1. Configuração da Câmera

Edite o arquivo de configuração YAML (ex: `config/stream_rtdetr_cam63.yaml`):

```yaml
cameras:
  - id: CAM063
    uri: "rtsp://usuario:senha@ip:porta/caminho"

out_dir: "./out"
save_video: true
video_fps: 1
```

### 2. Documentação Completa dos Parâmetros YAML

Veja a seção [**📋 Referência Completa de Parâmetros YAML**](#-referência-completa-de-parâmetros-yaml) abaixo para documentação detalhada de todos os parâmetros.

### 3. Configuração de ROI

Crie arquivos JSON com os polígonos de ROI em `rois/`:

```json
{
  "roi_epi_on": {
    "resolution": [1920, 1080],
    "polygons": [
      [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    ]
  }
}
```

### 4. Configuração de EPIs por ROI

No arquivo YAML da câmera, configure os EPIs necessários por ROI:

```yaml
roi_ppe_config:
  roi_epi_on:
    required_ppe:
      - helmet_white
      - vest
```

### 5. Configuração de Alertas

Crie o arquivo `db_config.env`:

```env
# Banco de Dados
DB_TYPE=mysql
DB_HOST=localhost
DB_PORT=3306
DB_USER=seu_usuario
DB_PASSWORD=sua_senha
DB_NAME=egtc_alerts

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Telegram (opcional)
TELEGRAM_TOKEN=seu_token
TELEGRAM_CHAT_ID=seu_chat_id

# Timezone
TIMEZONE_OFFSET_HOURS=-3.0

# Salvamento de Imagens
SAVE_ALERT_IMAGES=true
SAVE_CROP_ONLY=true
CROPS_DIR=crops
```

### 6. Configuração de Alertas no YAML

No arquivo YAML da câmera:

```yaml
# Sistema de Alertas
enable_alerts: true
show_alert_grid: true
alert_debounce_seconds: 15.0
alert_min_consecutive_frames: 20
alert_suppression_reset_seconds: 20.0
alert_hash_ttl_seconds: 60.0
alert_grid_size: 8
timezone_offset_hours: -3.0
save_alert_images: true
save_crop_only: true
crops_dir: crops
```

## 🚀 Uso

### Execução Básica

```bash
python3 pipeline_RETDETRX_YW.py \
  --config config/stream_rtdetr_cam63.yaml \
  --prompts config/ppe_prompts_rtdetr.yaml \
  --show-video
```

### Execução Completa com ROI e Alertas

```bash
python3 pipeline_RETDETRX_YW.py \
  --config config/stream_rtdetr_cam63.yaml \
  --prompts config/ppe_prompts_rtdetr.yaml \
  --roi rois/roi_cam63.json \
  --roi-polys roi_epi_on \
  --draw-roi \
  --show-video \
  --enable-alerts \
  --show-alert-grid \
  --alert-config db_config.env
```

### Parâmetros Principais

| Parâmetro | Descrição |
|-----------|-----------|
| `--config` | Arquivo YAML de configuração da câmera |
| `--prompts` | Arquivo YAML com prompts de EPIs |
| `--roi` | Arquivo JSON com definição de ROI |
| `--roi-polys` | Nome do polígono de ROI a usar |
| `--draw-roi` | Desenha ROI no vídeo |
| `--show-video` | Exibe vídeo em tempo real |
| `--no-save-video` | Não salva vídeo (apenas exibe) |
| `--enable-alerts` | Habilita envio de alertas (banco/Telegram) |
| `--show-alert-grid` | Exibe grid de alertas no vídeo |
| `--alert-config` | Caminho do arquivo de configuração de alertas |
| `--debug` | Modo debug com logs detalhados |
| `--show-rtdetr-boxes` | Mostra bounding boxes brutos do RT-DETR-X |

## 📁 Estrutura de Arquivos

```
egtc_detr/
├── pipeline_RETDETRX_YW.py    # Script principal
├── rtdetr_detector.py          # Detector de pessoas (RT-DETR-X)
├── ppe_detector.py             # Detector unificado de EPIs
├── yolo_world_ppe.py           # Implementação YOLO-World
├── tracker.py                  # Sistema de rastreamento
├── alerts.py                   # Sistema de alertas
├── utils.py                    # Funções utilitárias
├── ovd.py                      # Implementação OWL-V2 (opcional)
├── config/                     # Arquivos de configuração
│   ├── stream_rtdetr_cam63.yaml
│   └── ppe_prompts_rtdetr.yaml
├── rois/                       # Definições de ROI
│   └── roi_cam63.json
├── crops/                      # Imagens de alertas salvos
├── out/                        # Vídeos e métricas de saída
├── db_config.env               # Configuração de banco/Redis
└── requirements.txt            # Dependências Python
```

## 🎯 Funcionalidades Detalhadas

### Detecção de EPIs

O sistema detecta os seguintes EPIs (com base no prompt especificado com --prompts):
- Capacete (com detecção de cor: branco, vermelho, azul, amarelo, marrom, cinza)
- Colete refletivo
- Avental
- Luvas
- Protetor auricular

### Sistema de Alertas

1. **Confirmação de Violação**: 
   - Requer 20 frames consecutivos da mesma pessoa sem EPI
   - E pelo menos 15 segundos de violação
   - Ambos os critérios devem ser satisfeitos

2. **Supressão Espacial**: 
   - Evita alertas duplicados na mesma região (grid 8x8)
   - Reset automático após 20 segundos sem violação

3. **Integração Telegram**: 
   - Envia crop da pessoa sem EPI
   - Mensagem formatada com detalhes do evento
   - Timestamp no timezone local configurado

4. **Persistência**: 
   - Salva alertas no banco de dados
   - Armazena caminho da imagem salva
   - Suporta MySQL, PostgreSQL e Oracle

### Visualização

- **Bounding Boxes**:
  - Verde: Pessoa em conformidade
  - Amarelo: Violação detectada (aguardando confirmação) - "AVALIANDO"
  - Vermelho: Alerta enviado - "ALERTA"

- **Grid de Alertas**: 
  - Células vermelhas: Alertas enviados (suprimidas)
  - Células laranja: Violações ativas (aguardando confirmação)

- **Painel de EPIs**: 
  - Mostra status de cada EPI monitorado
  - Indica status do alerta (gerado, aguardando, etc.)

## 🔍 Troubleshooting

### Problema: Modelo não carrega

**Solução**: Verifique se o modelo está no diretório correto e se há espaço em disco suficiente. Os modelos serão baixados automaticamente na primeira execução.

### Problema: CUDA out of memory

**Solução**: Reduza o `imgsz` no arquivo YAML ou use um modelo menor (ex: `rtdetr-x.pt` ao invés de `rtdetr-l.pt`).

### Problema: Redis não conecta

**Solução**: 
1. Verifique se o Redis está rodando:
```bash
redis-cli ping
# Deve retornar: PONG
```

2. Verifique as configurações em `db_config.env`:
```env
REDIS_HOST=localhost
REDIS_PORT=6379
```

3. Se usar Redis remoto, verifique firewall e conectividade:
```bash
telnet <redis_host> 6379
```

4. Verifique logs do Redis:
```bash
sudo journalctl -u redis -f
```

### Problema: Banco de dados não conecta

**Solução**: 
1. Verifique as credenciais em `db_config.env`
2. Certifique-se de que o banco existe
3. A tabela será criada automaticamente na primeira execução

### Problema: Telegram não envia alertas

**Solução**:
1. Verifique se `TELEGRAM_TOKEN` e `TELEGRAM_CHAT_ID` estão corretos
2. Certifique-se de que `enable_alerts=true` no YAML ou use `--enable-alerts`
3. Verifique se o bot foi iniciado no Telegram

### Problema: Falsos positivos

**Solução**: Ajuste os parâmetros no YAML:
- Aumente `alert_min_consecutive_frames` (padrão: 20)
- Aumente `alert_debounce_seconds` (padrão: 15.0)
- Ajuste `rtdetr_conf` para filtrar detecções menos confiáveis

## 📊 Métricas

O sistema gera arquivos CSV com métricas em `out/metrics_CAM*.csv`:
- FPS de processamento
- Tempo de detecção (RT-DETR, PPE, tracking)
- Número de pessoas detectadas
- Número de violações

## 📋 Referência Completa de Parâmetros YAML

Esta seção documenta todos os parâmetros disponíveis no arquivo de configuração YAML (`config/stream_rtdetr_cam63.yaml`).

### Estrutura Básica

```yaml
cameras:
  - id: CAM063
    uri: "rtsp://usuario:senha@ip:porta/caminho"
```

### Parâmetros de Câmera

| Parâmetro | Tipo | Descrição | Valores Recomendados |
|-----------|------|-----------|---------------------|
| `cameras[].id` | string | Identificador único da câmera | Ex: `CAM063`, `CAM001` |
| `cameras[].uri` | string | URI da câmera (RTSP ou arquivo) | `rtsp://...` ou `/caminho/video.mp4` |

### Parâmetros de Saída

| Parâmetro | Tipo | Descrição | Performance | Qualidade |
|-----------|------|-----------|-------------|-----------|
| `out_dir` | string | Diretório para salvar vídeos e métricas | `"./out"` | `"./out"` |
| `save_video` | boolean | Salvar vídeo processado | `false` (mais rápido) | `true` (gravação) |
| `video_fps` | float | FPS de gravação do vídeo | `0.5` (menos espaço) | `1.0` (padrão) |

**Recomendações:**
- **Performance**: `save_video: false` se não precisar gravar
- **Qualidade**: `video_fps: 1.0` para capturar eventos importantes

### Parâmetros de Amostragem

| Parâmetro | Tipo | Descrição | Performance | Qualidade |
|-----------|------|-----------|-------------|-----------|
| `target_fps` | float | FPS alvo de processamento | `0.5` (menos carga) | `1.0` (padrão) |

**Recomendações:**
- **Performance**: `0.5` para reduzir carga (1 frame a cada 2 segundos)
- **Qualidade**: `1.0` para detecção mais responsiva

### Parâmetros RT-DETR-X (Detecção de Pessoas)

| Parâmetro | Tipo | Descrição | Performance | Qualidade |
|-----------|------|-----------|-------------|-----------|
| `rtdetr_weights` | string | Modelo RT-DETR-X | `rtdetr-m.pt` (rápido) | `rtdetr-l.pt` (preciso) |
| `rtdetr_imgsz` | int | Tamanho de processamento | `640` (rápido) | `1280` (preciso) |
| `rtdetr_conf` | float | Threshold de confiança | `0.4` (mais detecções) | `0.5` (menos falsos) |
| `rtdetr_iou` | float | Threshold IoU para NMS | `0.45` (padrão) | `0.45` (padrão) |

**Modelos disponíveis:**
- `rtdetr-x.pt`: Extra Large (mais preciso, mais lento)
- `rtdetr-l.pt`: Large (balanceado) ⭐ **Recomendado**
- `rtdetr-m.pt`: Medium (mais rápido, menos preciso)

**Recomendações:**
- **Performance**: `rtdetr-m.pt` + `imgsz: 640` + `conf: 0.4`
- **Qualidade**: `rtdetr-l.pt` + `imgsz: 1280` + `conf: 0.5`

### Filtros RT-DETR-X (Validação de Detecções)

| Parâmetro | Tipo | Descrição | Performance | Qualidade |
|-----------|------|-----------|-------------|-----------|
| `rtdetr_min_area` | float | Área mínima relativa (0-1) | `0.0001` (mais detecções) | `0.0005` (filtra pequenos) |
| `rtdetr_max_area` | float | Área máxima relativa (0-1) | `0.8` (padrão) | `0.8` (padrão) |
| `rtdetr_min_aspect_ratio` | float | Aspect ratio mínimo (altura/largura) | `0.25` (mais permissivo) | `0.4` (filtra largos) |
| `rtdetr_max_aspect_ratio` | float | Aspect ratio máximo | `6.0` (mais permissivo) | `3.5` (filtra cones) |
| `rtdetr_min_height_px` | int | Altura mínima em pixels | `20` (mais detecções) | `50` (filtra pequenos) |
| `rtdetr_min_width_px` | int | Largura mínima em pixels | `10` (mais detecções) | `20` (filtra estreitos) |
| `rtdetr_disable_filters` | boolean | Desabilita todos os filtros | `false` (recomendado) | `false` (recomendado) |

**Recomendações:**
- **Performance**: Valores mais permissivos para detectar mais pessoas
- **Qualidade**: Valores mais restritivos para filtrar falsos positivos (cones, objetos)

**Exemplo para filtrar cones:**
```yaml
rtdetr_max_aspect_ratio: 3.5    # Cones são muito alongados (> 4.0)
rtdetr_min_area: 0.0005         # Filtra objetos muito pequenos
rtdetr_min_height_px: 50        # Filtra objetos muito baixos
```

### Parâmetros do Detector de EPIs

| Parâmetro | Tipo | Descrição | Valores |
|-----------|------|-----------|---------|
| `ppe_detector` | string | Tipo de detector | `"yolo-world"` ou `"owl-v2"` |

**Recomendações:**
- **YOLO-World**: Mais rápido, boa precisão para objetos grandes
- **OWL-V2**: Mais preciso para roupas e objetos pequenos, mais lento

### Parâmetros YOLO-World

| Parâmetro | Tipo | Descrição | Performance | Qualidade |
|-----------|------|-----------|-------------|-----------|
| `yw_model` | string | Modelo YOLO-World | `yolov8s-world.pt` (rápido) | `yolov8m-world.pt` (preciso) |
| `yw_fp16` | boolean | Usar precisão FP16 (CUDA) | `true` (mais rápido) | `true` (recomendado) |
| `yw_score_thr` | float | Threshold de confiança | `0.20` (mais detecções) | `0.15` (mais sensível) |
| `yw_use_crop` | boolean | Processar crops individuais | `false` (mais rápido) | `true` (mais preciso) |
| `yw_crop_padding` | float | Padding relativo ao crop | `0.10` (menos contexto) | `0.20` (mais contexto) |
| `yw_min_crop_size` | int | Tamanho mínimo do crop (px) | `64` (menos processamento) | `32` (detecta pequenos) |
| `yw_imgsz` | int | Tamanho de processamento | `640` (rápido) | `1280` (preciso) |

**Modelos disponíveis:**
- `yolov8s-world.pt`: Small (mais rápido)
- `yolov8m-world.pt`: Medium (balanceado) ⭐ **Recomendado**
- `yolov8l-world.pt`: Large (mais preciso, mais lento)

**Recomendações:**
- **Performance**: `yolov8s-world.pt` + `imgsz: 640` + `use_crop: false`
- **Qualidade**: `yolov8m-world.pt` + `imgsz: 1280` + `use_crop: true` + `score_thr: 0.15`

**Nota sobre `yw_use_crop`:**
- `true`: Processa cada pessoa individualmente (melhor para objetos pequenos, mais lento)
- `false`: Processa frame completo (mais rápido, menos preciso para objetos pequenos)

### Parâmetros OWL-V2

| Parâmetro | Tipo | Descrição | Performance | Qualidade |
|-----------|------|-----------|-------------|-----------|
| `ovd_model` | string | Modelo OWL-V2 | `google/owlv2-base-patch16` (rápido) | `google/owlv2-large-patch16` (preciso) |
| `ovd_fp16` | boolean | Usar precisão FP16 | `true` (mais rápido) | `true` (recomendado) |
| `ovd_score_thr` | float | Threshold de confiança | `0.30` (menos falsos) | `0.26` (mais sensível) |
| `ovd_cache_dir` | string | Diretório de cache | `"./.hf"` (padrão) | `"./.hf"` (padrão) |
| `ovd_use_fast` | boolean | Usar processador rápido | `true` (recomendado) | `true` (recomendado) |
| `ovd_quantization_mode` | string | Modo de quantização | `"8bit"` (economia memória) | `"none"` (melhor qualidade) |

**Recomendações:**
- **Performance**: `base-patch16` + `fp16: true` + `quantization_mode: "8bit"`
- **Qualidade**: `large-patch16` + `fp16: true` + `quantization_mode: "none"`

### Parâmetros de Dispositivo

| Parâmetro | Tipo | Descrição | Valores |
|-----------|------|-----------|---------|
| `device` | string | Dispositivo de processamento | `"cuda"` (GPU) ou `"cpu"` |

**Recomendações:**
- Use `cuda` se tiver GPU NVIDIA (muito mais rápido)
- Use `cpu` apenas se não tiver GPU (muito mais lento)

### Parâmetros de Zonas (Heurísticas)

| Parâmetro | Tipo | Descrição | Performance | Qualidade |
|-----------|------|-----------|-------------|-----------|
| `head_ratio` | float | Proporção da cabeça (0-1) | `0.45` (padrão) | `0.45` (padrão) |
| `chest_min_ratio` | float | Início da zona do peito (0-1) | `0.25` (mais área) | `0.25` (mais área) |
| `chest_max_ratio` | float | Fim da zona do peito (0-1) | `0.85` (mais área) | `0.85` (mais área) |

**Explicação:**
- `head_ratio: 0.45`: Cabeça ocupa 0-45% da altura do bounding box
- `chest_min_ratio: 0.25` a `chest_max_ratio: 0.85`: Peito/torso ocupa 25-85% da altura

**Recomendações:**
- **Detecção de capacetes**: Ajuste `head_ratio` se necessário (padrão: `0.45`)
- **Detecção de coletes/aventais**: Amplie `chest_max_ratio` para `0.85` (mais área de busca)

### Parâmetros de Debounce

| Parâmetro | Tipo | Descrição | Performance | Qualidade |
|-----------|------|-----------|-------------|-----------|
| `debounce_seconds` | float | Tempo de debounce para violações | `5.0` (mais responsivo) | `8.0` (menos falsos) |

**Recomendações:**
- **Performance**: `5.0` para resposta mais rápida
- **Qualidade**: `8.0` para reduzir falsos positivos

### Parâmetros de Tracking (ByteTrack)

| Parâmetro | Tipo | Descrição | Performance | Qualidade |
|-----------|------|-----------|-------------|-----------|
| `track_thresh` | float | Threshold mínimo para criar track | `0.25` (mais tracks) | `0.25` (padrão) |
| `match_thresh` | float | Threshold IoU para associar detecções | `0.5` (mais re-ID) | `0.3` (menos re-ID) |
| `track_buffer` | int | Frames para manter track perdido | `30` (menos memória) | `60` (mais persistência) |
| `track_iou_thresh` | float | IoU para fallback tracker | `0.5` (mais re-ID) | `0.3` (menos re-ID) |
| `track_max_age` | int | Frames máximos sem detecção | `15` (menos memória) | `30` (mais persistência) |

**Recomendações:**
- **Performance**: Valores menores para menos uso de memória
- **Qualidade**: Valores maiores para melhor persistência de IDs (menos re-identificação)

**Explicação:**
- `match_thresh` menor = mais permissivo = menos re-ID (recomendado: `0.3`)
- `track_buffer` maior = mantém track por mais tempo = mais persistência (recomendado: `60`)

### Parâmetros de Sistema de Alertas

| Parâmetro | Tipo | Descrição | Performance | Qualidade |
|-----------|------|-----------|-------------|-----------|
| `enable_alerts` | boolean | Habilita envio de alertas | `false` (sem overhead) | `true` (funcional) |
| `show_alert_grid` | boolean | Exibe grid visual | `false` (sem overhead) | `true` (visualização) |
| `alert_debounce_seconds` | float | Tempo mínimo para confirmar violação | `10.0` (mais rápido) | `15.0` (menos falsos) |
| `alert_min_consecutive_frames` | int | Frames consecutivos mínimos | `10` (mais rápido) | `20` (menos falsos) |
| `alert_suppression_reset_seconds` | float | Tempo para resetar supressão | `15.0` (mais alertas) | `20.0` (menos spam) |
| `alert_hash_ttl_seconds` | float | TTL do hash de violação | `30.0` (menos memória) | `60.0` (menos duplicatas) |
| `alert_grid_size` | int | Tamanho do grid (NxN) | `4` (menos células) | `8` (mais granular) |
| `timezone_offset_hours` | float | Offset do timezone | `-3.0` (GMT-3) | Ajustar conforme localização |
| `save_alert_images` | boolean | Salvar imagens de alertas | `false` (sem I/O) | `true` (evidências) |
| `save_crop_only` | boolean | Salvar apenas crop (não frame completo) | `true` (menos espaço) | `true` (recomendado) |
| `crops_dir` | string | Diretório para salvar imagens | `"crops"` (padrão) | `"crops"` (padrão) |

**Recomendações:**
- **Performance**: Desabilite `save_alert_images` se não precisar
- **Qualidade**: Use `alert_min_consecutive_frames: 20` + `alert_debounce_seconds: 15.0` para reduzir falsos positivos

**Lógica de Confirmação:**
- Alerta é gerado quando: `frames >= alert_min_consecutive_frames` **E** `tempo >= alert_debounce_seconds`
- Ambos os critérios devem ser satisfeitos (lógica E)

### Parâmetros de Métricas

| Parâmetro | Tipo | Descrição | Performance | Qualidade |
|-----------|------|-----------|-------------|-----------|
| `metrics_overlay` | boolean | Mostrar métricas no vídeo | `false` (sem overhead) | `true` (monitoramento) |
| `metrics_csv` | boolean | Salvar métricas em CSV | `false` (sem I/O) | `true` (análise) |
| `metrics_print_every` | int | Imprimir métricas a cada N frames | `60` (menos logs) | `30` (mais frequente) |

**Recomendações:**
- **Performance**: Desabilite métricas se não precisar
- **Qualidade**: Mantenha habilitado para monitoramento

### Parâmetros de ROI e EPIs

| Parâmetro | Tipo | Descrição | Exemplo |
|-----------|------|-----------|---------|
| `roi_ppe_config` | dict | Mapeia ROI para EPIs obrigatórios | Ver exemplo abaixo |

**Exemplo:**
```yaml
roi_ppe_config:
  roi_epi_on:  # Nome do ROI no JSON
    - helmet        # Aceita qualquer capacete
    - helmet_white  # Exige capacete branco
    - vest          # Exige colete refletivo
    - gloves        # Exige luvas
```

**EPIs disponíveis:**
- `helmet`: Capacete (qualquer cor)
- `helmet_white`, `helmet_red`, `helmet_blue`, `helmet_yellow`, `helmet_brown`, `helmet_gray`: Capacete de cor específica
- `vest`: Colete refletivo
- `apron`: Avental
- `gloves`: Luvas
- `ear_protection`: Protetor auricular

### Configurações Recomendadas por Cenário

#### 🚀 Máxima Performance (GPU potente, menos precisão)
```yaml
rtdetr_weights: "rtdetr-m.pt"
rtdetr_imgsz: 640
ppe_detector: "yolo-world"
yw_model: "yolov8s-world.pt"
yw_imgsz: 640
yw_use_crop: false
target_fps: 0.5
save_video: false
save_alert_images: false
```

#### ⚖️ Balanceado (Recomendado)
```yaml
rtdetr_weights: "rtdetr-l.pt"
rtdetr_imgsz: 1280
ppe_detector: "yolo-world"
yw_model: "yolov8m-world.pt"
yw_imgsz: 1280
yw_use_crop: true
yw_score_thr: 0.15
target_fps: 1.0
alert_min_consecutive_frames: 20
alert_debounce_seconds: 15.0
```

#### 🎯 Máxima Qualidade (GPU potente, máxima precisão)
```yaml
rtdetr_weights: "rtdetr-l.pt"
rtdetr_imgsz: 1280
ppe_detector: "owl-v2"
ovd_model: "google/owlv2-large-patch16"
yw_imgsz: 1280
yw_use_crop: true
yw_score_thr: 0.15
target_fps: 1.0
alert_min_consecutive_frames: 20
alert_debounce_seconds: 15.0
rtdetr_conf: 0.5
rtdetr_max_aspect_ratio: 3.5
```

## 🔐 Segurança

⚠️ **IMPORTANTE**: 
- O arquivo `db_config.env` contém credenciais sensíveis
- **NÃO** commitado no Git.... criar localmemnte

## 📝 Licença

[Especificar licença do projeto]

## 👥 Contribuidores

[Lista de contribuidores]

## 📞 Suporte

Para suporte, entre em contato: contato@platformbuilders.io

---

**Desenvolvido por Platform Builders** - https://paltformbuilders.io
