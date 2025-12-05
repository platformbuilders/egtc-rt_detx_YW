#!/bin/bash
# Script para verificar dependências do sistema (apt-get)
# Lista o que está instalado, o que precisa ser instalado e o que pode ser atualizado

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APT_REQUIREMENTS_FILE="${SCRIPT_DIR}/apt_requirements.txt"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Verificando dependências do sistema"
echo "=========================================="
echo ""

# Verifica se o arquivo existe
if [ ! -f "$APT_REQUIREMENTS_FILE" ]; then
    echo -e "${RED}❌ Erro: Arquivo apt_requirements.txt não encontrado em $APT_REQUIREMENTS_FILE${NC}"
    exit 1
fi

# Verifica se dpkg está disponível
if ! command -v dpkg &> /dev/null; then
    echo -e "${RED}❌ Erro: dpkg não encontrado. Este script requer um sistema baseado em Debian/Ubuntu${NC}"
    exit 1
fi

# Arrays para armazenar resultados
INSTALLED=()
NOT_INSTALLED=()
OUTDATED=()
UNKNOWN=()

# Cache de pacotes instalados (carregado uma vez)
INSTALLED_PACKAGES=""

# Cache de pacotes atualizáveis (carregado uma vez)
UPGRADABLE_PACKAGES=""

# Carrega cache de pacotes instalados uma vez
load_installed_packages() {
    echo -ne "   Carregando lista de pacotes instalados..."
    INSTALLED_PACKAGES=$(dpkg -l 2>/dev/null | awk '/^ii/ {print $2}')
    echo -ne "\r                                                              \r"
}

# Carrega lista de pacotes atualizáveis uma vez (apenas se tiver permissões)
load_upgradable_list() {
    if [ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null; then
        echo -ne "   Carregando lista de pacotes atualizáveis..."
        UPGRADABLE_PACKAGES=$(apt list --upgradable 2>/dev/null | grep -E "^[^/]+/" | cut -d'/' -f1 || echo "")
        echo -ne "\r                                                              \r"
    else
        # Sem sudo, não verifica atualizações (muito lento)
        UPGRADABLE_PACKAGES=""
        echo "   (pulando verificação de atualizações - use sudo para verificar)"
    fi
}

# Função para verificar se um pacote está instalado
check_package() {
    local package=$1
    
    # Remove comentários inline e espaços
    package=$(echo "$package" | sed 's/#.*$//' | xargs)
    
    # Pula linhas vazias
    if [ -z "$package" ]; then
        return 2
    fi
    
    # Verifica se o pacote está instalado (usando cache)
    if echo "$INSTALLED_PACKAGES" | grep -qE "^${package}$"; then
        # Verifica se há atualizações disponíveis (apenas se tiver lista carregada)
        if [ -n "$UPGRADABLE_PACKAGES" ] && echo "$UPGRADABLE_PACKAGES" | grep -qE "^${package}$"; then
            OUTDATED+=("$package")
            return 1
        else
            INSTALLED+=("$package")
            return 0
        fi
    else
        # Verifica se o pacote existe nos repositórios (cache do apt-cache)
        # Usa timeout para evitar travamento
        if timeout 2 apt-cache show "$package" 2>/dev/null | grep -q "^Package:"; then
            NOT_INSTALLED+=("$package")
            return 2
        else
            UNKNOWN+=("$package")
            return 3
        fi
    fi
}

# Tenta atualizar lista de pacotes disponíveis (opcional, não crítico)
if [ "$EUID" -eq 0 ]; then
    echo -e "${BLUE}🔄 Atualizando lista de pacotes disponíveis...${NC}"
    apt update > /dev/null 2>&1 && echo -e "${GREEN}✓ Lista de pacotes atualizada${NC}" || echo -e "${YELLOW}⚠️  Não foi possível atualizar lista${NC}"
else
    echo -e "${BLUE}ℹ️  Executando verificação sem atualizar lista de pacotes (use sudo para atualizar)${NC}"
fi
echo ""

# Lê o arquivo e verifica cada pacote
echo -e "${BLUE}📦 Verificando pacotes...${NC}"
echo ""

# Carrega caches uma vez
load_installed_packages
load_upgradable_list

PACKAGE_COUNT=0
# Lê todos os pacotes do arquivo primeiro (mais eficiente)
# Ignora linhas que começam com #OPCIONAL (pacotes opcionais)
PACKAGES_TO_CHECK=$(grep -v '^#' "$APT_REQUIREMENTS_FILE" | grep -v '^#OPCIONAL' | grep -v '^$' | sed 's/#.*$//' | xargs)

if [ -z "$PACKAGES_TO_CHECK" ]; then
    echo -e "${YELLOW}⚠️  Nenhum pacote encontrado no arquivo apt_requirements.txt${NC}"
    exit 1
fi

# Processa cada pacote
TOTAL_PACKAGES=$(echo $PACKAGES_TO_CHECK | wc -w)
echo "   Verificando $TOTAL_PACKAGES pacotes..."
for package in $PACKAGES_TO_CHECK; do
    PACKAGE_COUNT=$((PACKAGE_COUNT + 1))
    # Mostra progresso a cada 10 pacotes
    if [ $((PACKAGE_COUNT % 10)) -eq 0 ] || [ $PACKAGE_COUNT -eq $TOTAL_PACKAGES ]; then
        echo -ne "\r   Progresso: $PACKAGE_COUNT/$TOTAL_PACKAGES pacotes verificados..."
    fi
    check_package "$package" || true  # Continua mesmo se houver erro
done
echo -ne "\r                                                              \r"
echo ""

echo ""

if [ $PACKAGE_COUNT -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Nenhum pacote encontrado no arquivo apt_requirements.txt${NC}"
    echo "   Verifique se o arquivo está no formato correto"
    exit 1
fi

# Mostra resultados
echo "=========================================="
echo "RESULTADO DA VERIFICAÇÃO"
echo "=========================================="
echo ""

# Pacotes instalados e atualizados
if [ ${#INSTALLED[@]} -gt 0 ]; then
    echo -e "${GREEN}✅ Pacotes instalados e atualizados (${#INSTALLED[@]}):${NC}"
    for pkg in "${INSTALLED[@]}"; do
        version=$(dpkg -l 2>/dev/null | awk -v pkg="$pkg" '$2 == pkg {print $3; exit}')
        echo -e "   ${GREEN}✓${NC} $pkg (versão: $version)"
    done
    echo ""
fi

# Pacotes instalados mas desatualizados
if [ ${#OUTDATED[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Pacotes instalados mas desatualizados (${#OUTDATED[@]}):${NC}"
    for pkg in "${OUTDATED[@]}"; do
        current_version=$(dpkg -l 2>/dev/null | awk -v pkg="$pkg" '$2 == pkg {print $3; exit}')
        echo -e "   ${YELLOW}⚠${NC} $pkg (atual: $current_version) - ${YELLOW}atualização disponível${NC}"
    done
    echo ""
fi

# Pacotes não instalados
if [ ${#NOT_INSTALLED[@]} -gt 0 ]; then
    echo -e "${RED}❌ Pacotes não instalados (${#NOT_INSTALLED[@]}):${NC}"
    for pkg in "${NOT_INSTALLED[@]}"; do
        echo -e "   ${RED}✗${NC} $pkg"
    done
    echo ""
fi

# Pacotes não encontrados nos repositórios
if [ ${#UNKNOWN[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Pacotes não encontrados nos repositórios (${#UNKNOWN[@]}):${NC}"
    for pkg in "${UNKNOWN[@]}"; do
        echo -e "   ${YELLOW}?${NC} $pkg - ${YELLOW}pode precisar de repositório adicional${NC}"
    done
    echo ""
fi

# Resumo
echo "=========================================="
echo "RESUMO"
echo "=========================================="
TOTAL=$(( ${#INSTALLED[@]} + ${#OUTDATED[@]} + ${#NOT_INSTALLED[@]} + ${#UNKNOWN[@]} ))
echo -e "Total de pacotes verificados: ${BLUE}$TOTAL${NC}"
echo -e "${GREEN}✅ Instalados e atualizados: ${#INSTALLED[@]}${NC}"
echo -e "${YELLOW}⚠️  Instalados mas desatualizados: ${#OUTDATED[@]}${NC}"
echo -e "${RED}❌ Não instalados: ${#NOT_INSTALLED[@]}${NC}"
if [ ${#UNKNOWN[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Não encontrados: ${#UNKNOWN[@]}${NC}"
fi
echo ""

# Comandos sugeridos
if [ ${#NOT_INSTALLED[@]} -gt 0 ] || [ ${#OUTDATED[@]} -gt 0 ]; then
    echo "=========================================="
    echo "AÇÕES SUGERIDAS"
    echo "=========================================="
    echo ""
    
    if [ ${#NOT_INSTALLED[@]} -gt 0 ]; then
        echo -e "${BLUE}Para instalar pacotes faltantes:${NC}"
        echo "sudo apt install -y ${NOT_INSTALLED[*]}"
        echo ""
    fi
    
    if [ ${#OUTDATED[@]} -gt 0 ]; then
        echo -e "${BLUE}Para atualizar pacotes desatualizados:${NC}"
        echo "sudo apt upgrade -y ${OUTDATED[*]}"
        echo ""
        echo -e "${BLUE}Ou atualizar todos os pacotes do sistema:${NC}"
        echo "sudo apt update && sudo apt upgrade -y"
        echo ""
    fi
    
    if [ ${#NOT_INSTALLED[@]} -gt 0 ] && [ ${#OUTDATED[@]} -gt 0 ]; then
        echo -e "${BLUE}Para instalar e atualizar tudo de uma vez:${NC}"
        echo "sudo apt update && sudo apt install -y ${NOT_INSTALLED[*]} && sudo apt upgrade -y ${OUTDATED[*]}"
        echo ""
    fi
fi

# Status final
if [ ${#NOT_INSTALLED[@]} -eq 0 ] && [ ${#OUTDATED[@]} -eq 0 ] && [ ${#UNKNOWN[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ Todas as dependências estão instaladas e atualizadas!${NC}"
    exit 0
elif [ ${#UNKNOWN[@]} -gt 0 ] && [ ${#NOT_INSTALLED[@]} -eq 0 ] && [ ${#OUTDATED[@]} -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Alguns pacotes não foram encontrados, mas os demais estão OK${NC}"
    exit 0
else
    echo -e "${RED}❌ Algumas dependências precisam ser instaladas ou atualizadas${NC}"
    exit 1
fi

