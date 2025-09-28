/**
 * NEXUS AI - Advanced Educational Platform JavaScript
 * Handles real-time updates, animations, and interactive features
 */

class NexusAI {
    constructor() {
        this.charts = {};
        this.updateInterval = null;
        this.animationQueue = [];
        this.isInitialized = false;
        
        this.init();
    }

    init() {
        if (this.isInitialized) return;
        
        console.log('🚀 Initializing NEXUS AI Platform...');
        
        this.setupEventListeners();
        this.initializeAnimations();
        this.startPerformanceMonitoring();
        
        this.isInitialized = true;
        console.log('✅ NEXUS AI Platform initialized successfully');
    }

    setupEventListeners() {
        // Page load animations
        document.addEventListener('DOMContentLoaded', () => {
            this.animatePageLoad();
        });

        // Intersection Observer for scroll animations
        this.setupScrollAnimations();

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });

        // Mouse interactions
        this.setupMouseInteractions();
    }

    animatePageLoad() {
        const cards = document.querySelectorAll('.glass-card, .metric-card, .nexus-card');
        
        cards.forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(30px)';
            
            setTimeout(() => {
                card.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }

    setupScrollAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });

        document.querySelectorAll('.glass-card, .metric-card').forEach(el => {
            observer.observe(el);
        });
    }

    setupMouseInteractions() {
        // Parallax effect for background elements
        document.addEventListener('mousemove', (e) => {
            const mouseX = e.clientX / window.innerWidth;
            const mouseY = e.clientY / window.innerHeight;
            
            const floatingElements = document.querySelectorAll('.floating-animation');
            floatingElements.forEach((el, index) => {
                const speed = (index + 1) * 0.5;
                const x = (mouseX - 0.5) * speed * 20;
                const y = (mouseY - 0.5) * speed * 20;
                
                el.style.transform = `translate(${x}px, ${y}px)`;
            });
        });

        // Card hover effects
        document.querySelectorAll('.glass-card, .metric-card').forEach(card => {
            card.addEventListener('mouseenter', (e) => {
                this.addHoverGlow(e.target);
            });
            
            card.addEventListener('mouseleave', (e) => {
                this.removeHoverGlow(e.target);
            });
        });
    }

    addHoverGlow(element) {
        element.style.boxShadow = '0 20px 40px rgba(102, 126, 234, 0.2)';
        element.style.transform = 'translateY(-4px) scale(1.02)';
    }

    removeHoverGlow(element) {
        element.style.boxShadow = '';
        element.style.transform = '';
    }

    handleKeyboardShortcuts(e) {
        // Ctrl/Cmd + K for search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            this.openSearch();
        }
        
        // Escape to close modals
        if (e.key === 'Escape') {
            this.closeModals();
        }
    }

    initializeAnimations() {
        // Stagger animations for multiple elements
        this.staggerAnimation('.ai-insight', 'slideInUp', 200);
        this.staggerAnimation('.recommendation-card', 'fadeInScale', 150);
    }

    staggerAnimation(selector, animationClass, delay) {
        const elements = document.querySelectorAll(selector);
        elements.forEach((el, index) => {
            setTimeout(() => {
                el.classList.add(animationClass);
            }, index * delay);
        });
    }

    // Real-time metrics updates
    startLiveMetricsUpdate(studentId) {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        this.updateInterval = setInterval(() => {
            this.updateLiveMetrics(studentId);
        }, 3000);
    }

    async updateLiveMetrics(studentId) {
        try {
            const response = await fetch(`/api/student/${studentId}/live_metrics`);
            const data = await response.json();
            
            this.animateMetricUpdate('session-time', data.current_session_time + 'm');
            this.animateMetricUpdate('concepts-today', data.concepts_learned_today);
            this.animateMetricUpdate('accuracy-rate', data.accuracy_rate.toFixed(1) + '%');
            this.animateMetricUpdate('focus-score', data.focus_score.toFixed(0));
            this.animateMetricUpdate('ai-confidence', data.ai_confidence.toFixed(1) + '%');
            this.animateMetricUpdate('milestone-progress', data.next_milestone_progress.toFixed(0) + '%');
            
        } catch (error) {
            console.error('Error updating live metrics:', error);
        }
    }

    animateMetricUpdate(elementId, newValue) {
        const element = document.getElementById(elementId);
        if (!element) return;

        // Add update animation
        element.style.transform = 'scale(1.1)';
        element.style.color = '#10b981';
        
        setTimeout(() => {
            element.textContent = newValue;
            element.style.transform = 'scale(1)';
            element.style.color = '';
        }, 150);
    }

    // Chart management
    createAdvancedChart(canvasId, config) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        // Destroy existing chart
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        // Enhanced chart configuration
        const enhancedConfig = {
            ...config,
            options: {
                ...config.options,
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    ...config.options?.plugins,
                    legend: {
                        display: false,
                        ...config.options?.plugins?.legend
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeInOutQuart'
                }
            }
        };

        this.charts[canvasId] = new Chart(ctx, enhancedConfig);
        return this.charts[canvasId];
    }

    // Progress ring animation
    animateProgressRing(elementId, percentage) {
        const ring = document.querySelector(`#${elementId} .progress`);
        if (!ring) return;

        const circumference = 188.5;
        const offset = circumference - (percentage / 100) * circumference;
        
        ring.style.strokeDashoffset = offset;
    }

    // Skill bar animation
    animateSkillBar(elementId, percentage) {
        const bar = document.querySelector(`#${elementId} .skill-progress`);
        if (!bar) return;

        setTimeout(() => {
            bar.style.width = percentage + '%';
        }, 100);
    }

    // Notification system
    showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${this.getNotificationIcon(type)} mr-2"></i>
                ${message}
            </div>
        `;

        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);

        // Remove after duration
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, duration);
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    // Performance monitoring
    startPerformanceMonitoring() {
        // Monitor page performance
        if ('performance' in window) {
            window.addEventListener('load', () => {
                const perfData = performance.getEntriesByType('navigation')[0];
                console.log(`📊 Page load time: ${perfData.loadEventEnd - perfData.loadEventStart}ms`);
            });
        }

        // Monitor memory usage (if available)
        if ('memory' in performance) {
            setInterval(() => {
                const memory = performance.memory;
                if (memory.usedJSHeapSize > memory.jsHeapSizeLimit * 0.9) {
                    console.warn('⚠️ High memory usage detected');
                }
            }, 30000);
        }
    }

    // Utility methods
    openSearch() {
        console.log('🔍 Opening search...');
        // Implement search functionality
    }

    closeModals() {
        const modals = document.querySelectorAll('.modal.show');
        modals.forEach(modal => {
            modal.classList.remove('show');
        });
    }

    // Cleanup
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });

        this.charts = {};
        this.isInitialized = false;
    }
}

// Global NEXUS AI instance
window.nexusAI = new NexusAI();

// Utility functions for global use
window.selectStudent = function(studentId) {
    const card = event.currentTarget;
    card.style.transform = 'scale(0.95)';
    card.style.opacity = '0.8';
    
    nexusAI.showNotification(`Loading ${studentId}'s dashboard...`, 'info', 1000);
    
    setTimeout(() => {
        window.location.href = `/student/${studentId}`;
    }, 200);
};

window.startStudentDashboard = function(studentId) {
    nexusAI.startLiveMetricsUpdate(studentId);
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NexusAI;
}
