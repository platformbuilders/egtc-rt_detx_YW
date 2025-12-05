#!/bin/bash
# Script para instalar dependências do sistema (apt-get)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APT_REQUIREMENTS_FILE="${SCRIPT_DIR}/apt_requirements.txt"

echo "=========================================="
echo "Instalando dependências do sistema"
echo "=========================================="

# Verifica se está rodando como root ou com sudo
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Este script precisa de privilégios de administrador"
    echo "   Execute com: sudo $0"
    exit 1
fi

# Atualiza lista de pacotes
echo "🔄 Atualizando lista de pacotes..."
apt update

# Lê o arquivo de requisitos e instala
if [ ! -f "$APT_REQUIREMENTS_FILE" ]; then
    echo "❌ Erro: Arquivo apt_requirements.txt não encontrado em $APT_REQUIREMENTS_FILE"
    exit 1
fi

echo "📦 Instalando pacotes do apt_requirements.txt..."
echo ""

# Filtra comentários e linhas vazias, depois instala
PACKAGES=$(grep -v '^#' "$APT_REQUIREMENTS_FILE" | grep -v '^$' | tr '\n' ' ')

if [ -z "$PACKAGES" ]; then
    echo "⚠️  Nenhum pacote encontrado para instalar"
    exit 0
fi

echo "Pacotes a serem instalados:"
echo "$PACKAGES" | tr ' ' '\n' | grep -v '^$' | sed 's/^/   - /'
echo ""

read -p "Continuar com a instalação? (s/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo "❌ Instalação cancelada"
    exit 0
fi

# Instala os pacotes
apt install -y $PACKAGES

echo ""
echo "✅ Instalação concluída!"
echo ""
echo "💡 Próximos passos:"
echo "   1. Configure o Redis: sudo systemctl start redis-server && sudo systemctl enable redis-server"
echo "   2. Crie o ambiente virtual: python3 -m venv egtc_detr_venv"
echo "   3. Instale dependências Python: pip install -r requirements.txt"

