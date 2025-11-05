#!/bin/bash

# KTCD_Aug Setup Script
# Automated setup for development and production environments

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_command() {
    if command -v $1 &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

# Main setup
main() {
    print_header "KTCD_Aug Setup Script"
    echo ""

    # Check prerequisites
    print_header "Checking Prerequisites"
    
    MISSING_DEPS=0
    
    if ! check_command python3; then
        print_error "Python 3 is required. Please install Python 3.12+"
        MISSING_DEPS=1
    else
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        echo -e "  Version: ${PYTHON_VERSION}"
    fi
    
    if ! check_command docker; then
        print_warning "Docker is recommended for easy deployment"
    fi
    
    if ! check_command docker-compose; then
        print_warning "Docker Compose is recommended for easy deployment"
    fi
    
    if ! check_command git; then
        print_warning "Git is recommended for version control"
    fi
    
    if [ $MISSING_DEPS -eq 1 ]; then
        print_error "Please install missing dependencies and try again"
        exit 1
    fi
    
    echo ""
    
    # Create virtual environment
    print_header "Setting Up Virtual Environment"
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
            python3 -m venv venv
            print_success "Virtual environment recreated"
        fi
    else
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    echo ""
    
    # Install dependencies
    print_header "Installing Dependencies"
    
    pip install --upgrade pip
    print_success "pip upgraded"
    
    pip install -r requirements.txt
    print_success "Dependencies installed"
    
    echo ""
    
    # Create .env file
    print_header "Configuring Environment"
    
    if [ -f ".env" ]; then
        print_warning ".env file already exists"
        read -p "Do you want to overwrite it? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cp .env.example .env
            print_success ".env file created"
        fi
    else
        cp .env.example .env
        print_success ".env file created"
    fi
    
    echo ""
    
    # Create necessary directories
    print_header "Creating Directories"
    
    mkdir -p logs
    print_success "logs/ directory created"
    
    mkdir -p data/generated_blogs
    print_success "data/generated_blogs/ directory created"
    
    mkdir -p models/checkpoints
    print_success "models/checkpoints/ directory created"
    
    mkdir -p backups
    print_success "backups/ directory created"
    
    echo ""
    
    # Docker setup
    print_header "Docker Setup"
    
    if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        read -p "Do you want to start Docker services now? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose up -d neo4j
            print_success "Neo4j started"
            
            echo "Waiting for Neo4j to be ready..."
            sleep 10
            
            print_success "Docker services started"
        else
            print_warning "You can start Docker services later with: docker-compose up -d"
        fi
    else
        print_warning "Docker not available. You'll need to set up Neo4j manually"
    fi
    
    echo ""
    
    # Final instructions
    print_header "Setup Complete!"
    
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo ""
    echo "1. Edit .env file with your configuration:"
    echo -e "   ${YELLOW}nano .env${NC}"
    echo ""
    echo "2. Start the application:"
    echo -e "   ${YELLOW}python nexus_app.py${NC}"
    echo ""
    echo "   Or with Docker:"
    echo -e "   ${YELLOW}docker-compose up -d${NC}"
    echo ""
    echo "3. Access the platform:"
    echo -e "   ${YELLOW}http://localhost:8080${NC}"
    echo ""
    echo "4. View documentation:"
    echo -e "   ${YELLOW}cat ULTIMATE_PROJECT_SUMMARY.md${NC}"
    echo ""
    echo -e "${BLUE}For more commands, run:${NC} ${YELLOW}make help${NC}"
    echo ""
    
    print_success "Setup completed successfully!"
}

# Run main function
main

