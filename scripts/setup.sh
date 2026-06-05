#!/bin/bash
# ================================================================
# TalentAI-X — Complete Setup Script
# Run this once to set everything up from scratch.
# ================================================================

set -e  # Exit on any error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║         TalentAI-X Setup Script             ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Step 1: Check prerequisites ──────────────────────────────────
echo -e "${YELLOW}[1/7] Checking prerequisites...${NC}"

command -v python3 >/dev/null 2>&1 || { echo -e "${RED}Python 3.12+ required${NC}"; exit 1; }
command -v node >/dev/null 2>&1 || { echo -e "${RED}Node.js 20+ required${NC}"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo -e "${RED}Docker required${NC}"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || command -v "docker compose" >/dev/null 2>&1 || { echo -e "${RED}Docker Compose required${NC}"; exit 1; }

echo -e "${GREEN}✓ All prerequisites found${NC}"

# ── Step 2: Environment file ──────────────────────────────────────
echo -e "${YELLOW}[2/7] Setting up environment...${NC}"

if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${YELLOW}⚠ .env created from template. Add your ANTHROPIC_API_KEY!${NC}"
    echo ""
    echo "  Edit .env and set:"
    echo "  ANTHROPIC_API_KEY=your_key_here"
    echo ""
    read -p "Press Enter after setting ANTHROPIC_API_KEY in .env..."
fi

# Validate API key is set
if grep -q "your_anthropic_api_key_here" .env 2>/dev/null; then
    echo -e "${RED}ERROR: ANTHROPIC_API_KEY not set in .env${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Environment configured${NC}"

# ── Step 3: Start infrastructure ─────────────────────────────────
echo -e "${YELLOW}[3/7] Starting infrastructure (PostgreSQL, Redis, ChromaDB)...${NC}"
docker compose up -d postgres redis chromadb
echo "Waiting for services to be ready..."
sleep 8

# Check health
docker compose ps postgres | grep -q "healthy" && echo -e "${GREEN}✓ PostgreSQL ready${NC}" || echo -e "${YELLOW}⚠ PostgreSQL starting...${NC}"
docker compose ps redis | grep -q "healthy" && echo -e "${GREEN}✓ Redis ready${NC}" || echo -e "${YELLOW}⚠ Redis starting...${NC}"

# ── Step 4: Backend Python setup ─────────────────────────────────
echo -e "${YELLOW}[4/7] Setting up Python backend...${NC}"
cd backend

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
echo -e "${GREEN}✓ NLTK data downloaded${NC}"

cd ..

# ── Step 5: Frontend Node setup ───────────────────────────────────
echo -e "${YELLOW}[5/7] Setting up frontend...${NC}"
cd frontend
npm install --silent
echo -e "${GREEN}✓ Node.js dependencies installed${NC}"
cd ..

# ── Step 6: Generate sample data ─────────────────────────────────
echo -e "${YELLOW}[6/7] Generating sample data...${NC}"
cd data
python3 generate_resumes.py
cd ..
echo -e "${GREEN}✓ Sample data ready${NC}"

# ── Step 7: Done ─────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗"
echo "║           Setup Complete! 🚀                ║"
echo "╚══════════════════════════════════════════════╝${NC}"
echo ""
echo "To start the full system:"
echo ""
echo "  Terminal 1 (Backend API):"
echo "    cd backend && source venv/bin/activate"
echo "    uvicorn app.main:app --reload --port 8000"
echo ""
echo "  Terminal 2 (Celery Worker):"
echo "    cd backend && source venv/bin/activate"
echo "    celery -A app.worker.celery_app worker --loglevel=info"
echo ""
echo "  Terminal 3 (Frontend):"
echo "    cd frontend && npm run dev"
echo ""
echo "  OR use Docker Compose for everything:"
echo "    docker compose up"
echo ""
echo "URLs:"
echo "  Frontend:  http://localhost:3000"
echo "  API:       http://localhost:8000"
echo "  Swagger:   http://localhost:8000/docs"
echo "  ChromaDB:  http://localhost:8001"
echo ""
echo "Default API key: dev_key_change_in_production"
echo ""
