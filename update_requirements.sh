#!/bin/bash
# Script para atualizar requirements.txt a partir do ambiente virtual atual

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/egtc_detr_venv"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

echo "=========================================="
echo "Atualizando requirements.txt"
echo "=========================================="

# Verifica se o ambiente virtual existe
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Erro: Ambiente virtual não encontrado em $VENV_DIR"
    echo "   Execute primeiro: python3 -m venv egtc_detr_venv"
    exit 1
fi

# Ativa o ambiente virtual
echo "📦 Ativando ambiente virtual..."
source "${VENV_DIR}/bin/activate"

# Verifica se pip está instalado
if ! command -v pip &> /dev/null; then
    echo "❌ Erro: pip não encontrado no ambiente virtual"
    exit 1
fi

# Faz backup do requirements.txt atual
if [ -f "$REQUIREMENTS_FILE" ]; then
    BACKUP_FILE="${REQUIREMENTS_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "💾 Fazendo backup do requirements.txt atual para: $BACKUP_FILE"
    cp "$REQUIREMENTS_FILE" "$BACKUP_FILE"
fi

# Gera novo requirements.txt
echo "🔄 Gerando novo requirements.txt..."
pip freeze > "$REQUIREMENTS_FILE"

# Conta quantas dependências foram geradas
PACKAGE_COUNT=$(grep -c "^[^#]" "$REQUIREMENTS_FILE" || echo "0")
echo "✅ requirements.txt atualizado com $PACKAGE_COUNT pacotes"

# Mostra algumas estatísticas
echo ""
echo "📊 Estatísticas:"
echo "   - Total de pacotes: $PACKAGE_COUNT"
echo "   - Arquivo: $REQUIREMENTS_FILE"

# Pergunta se quer ver as diferenças
if [ -f "$BACKUP_FILE" ]; then
    echo ""
    read -p "Deseja ver as diferenças? (s/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        echo ""
        echo "📋 Diferenças (novo vs antigo):"
        diff -u "$BACKUP_FILE" "$REQUIREMENTS_FILE" || echo "   (arquivos idênticos ou sem diferenças significativas)"
    fi
fi

echo ""
echo "✅ Concluído!"
echo ""
echo "💡 Próximos passos:"
echo "   1. Revise o requirements.txt gerado"
echo "   2. Remova pacotes desnecessários se houver"
echo "   3. Teste a instalação: pip install -r requirements.txt"

