#!/bin/bash

# MCP-SGR Deployment Script
# Supports blue/green deployment for zero-downtime updates

set -euo pipefail

# Configuration
ENVIRONMENT="${1:-staging}"
VERSION="${2:-latest}"
HEALTH_CHECK_URL="http://localhost:8080/health"
HEALTH_CHECK_TIMEOUT=60
ROLLBACK_TIMEOUT=300

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root. Consider using a dedicated deployment user."
    fi
}

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        staging|production)
            log_info "Deploying to $ENVIRONMENT environment"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running"
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        log_error "docker-compose is not installed"
        exit 1
    fi
    
    # Check if required files exist
    COMPOSE_FILE="docker-compose.${ENVIRONMENT}.yml"
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Pull latest images
pull_images() {
    log_info "Pulling latest images..."
    
    export VERSION
    docker-compose -f "docker-compose.${ENVIRONMENT}.yml" pull
    
    log_success "Images pulled successfully"
}

# Health check function
health_check() {
    local url=$1
    local timeout=${2:-60}
    local interval=5
    local elapsed=0
    
    log_info "Performing health check on $url"
    
    while [[ $elapsed -lt $timeout ]]; do
        if curl -f -s "$url" >/dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
        echo -n "."
    done
    
    log_error "Health check failed after ${timeout}s"
    return 1
}

# Deploy to staging
deploy_staging() {
    log_info "Deploying to staging environment..."
    
    export VERSION
    docker-compose -f docker-compose.staging.yml down || true
    docker-compose -f docker-compose.staging.yml up -d
    
    # Wait for services to be ready
    sleep 10
    
    # Health check
    if health_check "http://localhost:8081/health" 60; then
        log_success "Staging deployment completed successfully"
        
        # Run smoke tests
        run_smoke_tests "staging"
    else
        log_error "Staging deployment failed health check"
        return 1
    fi
}

# Blue/Green deployment for production
deploy_production() {
    log_info "Starting blue/green deployment to production..."
    
    # Check current active deployment
    CURRENT_ACTIVE=$(get_active_deployment)
    if [[ "$CURRENT_ACTIVE" == "blue" ]]; then
        TARGET_SLOT="green"
        CURRENT_SLOT="blue"
    else
        TARGET_SLOT="blue"
        CURRENT_SLOT="green"
    fi
    
    log_info "Current active: $CURRENT_SLOT, deploying to: $TARGET_SLOT"
    
    # Deploy to target slot
    export VERSION
    docker-compose -f docker-compose.production.yml up -d "mcp-sgr-${TARGET_SLOT}"
    
    # Wait for new deployment to be ready
    sleep 15
    
    # Health check on new deployment
    local target_port
    if [[ "$TARGET_SLOT" == "green" ]]; then
        target_port="8082"
    else
        target_port="8080"
    fi
    
    if health_check "http://localhost:${target_port}/health" 120; then
        log_success "New deployment ($TARGET_SLOT) is healthy"
        
        # Switch traffic to new deployment
        switch_traffic "$TARGET_SLOT"
        
        # Final health check on load balancer
        if health_check "http://localhost/health" 30; then
            log_success "Traffic successfully switched to $TARGET_SLOT"
            
            # Stop old deployment after grace period
            log_info "Stopping old deployment ($CURRENT_SLOT) after grace period..."
            sleep 30
            docker-compose -f docker-compose.production.yml stop "mcp-sgr-${CURRENT_SLOT}"
            
            log_success "Production deployment completed successfully"
        else
            log_error "Load balancer health check failed, rolling back..."
            rollback_deployment "$CURRENT_SLOT"
            return 1
        fi
    else
        log_error "New deployment health check failed"
        docker-compose -f docker-compose.production.yml stop "mcp-sgr-${TARGET_SLOT}"
        return 1
    fi
}

# Get currently active deployment slot
get_active_deployment() {
    # Check which container is receiving traffic from HAProxy
    # This is a simplified check - in production you'd query HAProxy stats
    if docker-compose -f docker-compose.production.yml ps mcp-sgr-blue | grep -q "Up"; then
        echo "blue"
    else
        echo "green"
    fi
}

# Switch traffic between blue/green deployments
switch_traffic() {
    local target_slot=$1
    
    log_info "Switching traffic to $target_slot deployment"
    
    # Update HAProxy configuration or use HAProxy API
    # This is a placeholder - implement actual traffic switching logic
    curl -X POST "http://localhost:8405/?action=switch_to_${target_slot}" || true
    
    log_success "Traffic switched to $target_slot"
}

# Rollback deployment
rollback_deployment() {
    local rollback_slot=$1
    
    log_warning "Rolling back to $rollback_slot deployment"
    
    # Switch traffic back
    switch_traffic "$rollback_slot"
    
    # Start the rollback slot if it's not running
    docker-compose -f docker-compose.production.yml up -d "mcp-sgr-${rollback_slot}"
    
    log_success "Rollback completed"
}

# Run smoke tests
run_smoke_tests() {
    local env=$1
    
    log_info "Running smoke tests for $env environment..."
    
    local base_url
    if [[ "$env" == "staging" ]]; then
        base_url="http://localhost:8081"
    else
        base_url="http://localhost"
    fi
    
    # Test health endpoint
    if ! curl -f -s "$base_url/health" >/dev/null; then
        log_error "Health endpoint test failed"
        return 1
    fi
    
    # Test docs endpoint
    if ! curl -f -s "$base_url/docs" >/dev/null; then
        log_error "Docs endpoint test failed"
        return 1
    fi
    
    # Test API endpoint (if auth is disabled for testing)
    # curl -f -s "$base_url/v1/schemas" >/dev/null || log_warning "API test skipped (auth required)"
    
    log_success "Smoke tests passed"
}

# Backup before deployment
backup_data() {
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_info "Creating backup before production deployment..."
        
        # Run backup script
        if [[ -f "scripts/backup.sh" ]]; then
            bash scripts/backup.sh
            log_success "Backup completed"
        else
            log_warning "Backup script not found"
        fi
    fi
}

# Send deployment notification
send_notification() {
    local status=$1
    local details=${2:-""}
    
    # Send notification to team (Slack, Discord, email, etc.)
    log_info "Sending deployment notification: $status"
    
    # Example: Slack webhook
    # curl -X POST -H 'Content-type: application/json' \
    #     --data "{\"text\":\"🚀 MCP-SGR $ENVIRONMENT deployment $status\n$details\"}" \
    #     "$SLACK_WEBHOOK_URL"
}

# Main deployment function
main() {
    log_info "Starting MCP-SGR deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    
    check_permissions
    validate_environment
    check_prerequisites
    
    # Create backup for production
    backup_data
    
    # Pull latest images
    pull_images
    
    # Deploy based on environment
    if [[ "$ENVIRONMENT" == "staging" ]]; then
        if deploy_staging; then
            send_notification "SUCCESS" "Staging deployment completed"
        else
            send_notification "FAILED" "Staging deployment failed"
            exit 1
        fi
    else
        if deploy_production; then
            send_notification "SUCCESS" "Production deployment completed"
        else
            send_notification "FAILED" "Production deployment failed"
            exit 1
        fi
    fi
    
    log_success "Deployment completed successfully! 🎉"
}

# Script usage
usage() {
    echo "Usage: $0 <environment> [version]"
    echo "  environment: staging or production"
    echo "  version: Docker image version (default: latest)"
    echo ""
    echo "Examples:"
    echo "  $0 staging"
    echo "  $0 production v1.2.3"
    echo "  $0 staging latest"
}

# Handle script arguments
if [[ $# -eq 0 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Run main function
main "$@"