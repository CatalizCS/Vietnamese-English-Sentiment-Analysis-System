// Initialize service worker globals
let backgroundService = null;
let pollInterval = null;

// Service worker setup
async function initializeServiceWorker() {
    try {
        if (!backgroundService) {
            backgroundService = new BackgroundService();
            await backgroundService.initialize();
        }
    } catch (error) {
        console.error('Service worker initialization failed:', error);
    }
}

// Service worker activation events
chrome.runtime.onInstalled.addListener(initializeServiceWorker);
chrome.runtime.onStartup.addListener(initializeServiceWorker);

// Cleanup on service worker update/unload
self.addEventListener('unload', () => {
    if (pollInterval) {
        clearInterval(pollInterval);
    }
});

class BackgroundService {
    constructor() {
        this.API_URL = 'http://workspace.tamais.me:7270';
        this.stats = {
            analyzed: 0,
            successful: 0
        };
        this.lastApiCheck = null;
        this.retryTimeout = 5000;
        this.maxRetries = 3;
        this.lastStatus = null;
        this.retryDelays = [1000, 2000, 4000];
        this.healthCheckInterval = 30000;
        this.initApiUrl();
        this.popupPorts = new Set();
        this.isInitialized = false;
        this.connectionAttempts = 0;
        this.maxConnectionAttempts = 5;
        this.messageQueue = [];
        this.portConnectionRetries = 0;
        this.maxPortRetries = 3;
        this.pendingRequests = new Map();
        this.failedAttempts = 0;
        this.maxFailedAttempts = 3;
        this.apiTimeout = 3000; // 3 seconds timeout
        this.dataCache = {
            lastUpdate: null,
            stats: null,
            status: null
        };
        this.dataRefreshInterval = 3000; // 3 seconds
    }

    async initialize() {
        if (this.isInitialized) return;
        await this.initApiUrl();
        this.setupMessageHandlers();
        await this.startApiMonitoring();
        this.startDataRefreshCycle();
        this.isInitialized = true;
    }

    startDataRefreshCycle() {
        setInterval(async () => {
            await this.refreshData();
        }, this.dataRefreshInterval);
    }

    async refreshData() {
        try {
            const status = await this.checkApiStatus();
            this.dataCache = {
                lastUpdate: new Date(),
                stats: this.stats,
                status: status,
                models: status.models || { vi: false, en: false },
                errors: status.error ? [status.error] : []
            };
            await this.broadcastUpdate();
            this.pruneDataCache(); // Remove old data
        } catch (error) {
            console.error('Data refresh failed:', error);
            this.dataCache.errors.push(error.message);
        }
    }

    pruneDataCache() {
        // Remove data older than 5 minutes
        const fiveMinutesAgo = Date.now() - (5 * 60 * 1000);
        this.dataCache.errors = this.dataCache.errors.filter(err =>
            err.timestamp > fiveMinutesAgo
        );
    }

    async broadcastUpdate() {
        // Notify all content scripts
        chrome.tabs.query({}, (tabs) => {
            tabs.forEach(tab => {
                chrome.tabs.sendMessage(tab.id, {
                    type: 'DATA_UPDATE',
                    data: this.dataCache
                }).catch(() => { });
            });
        });

        // Notify popup
        await this.updateAll();
    }

    async initApiUrl() {
        const { apiUrl } = await chrome.storage.local.get('apiUrl');
        this.API_URL = apiUrl || 'http://workspace.tamais.me:7270';
    }

    setupMessageHandlers() {
        chrome.runtime.onConnect.addListener((port) => {
            if (port.name === 'popup') {
                const portId = Date.now().toString();
                this.popupPorts.add(port);

                const pingInterval = setInterval(() => {
                    if (this.isValidPort(port)) {
                        try {
                            port.postMessage({ type: 'PING' });
                        } catch (e) {
                            clearInterval(pingInterval);
                        }
                    } else {
                        clearInterval(pingInterval);
                    }
                }, 5000);

                port.onDisconnect.addListener(() => {
                    clearInterval(pingInterval);
                    this.popupPorts.delete(port);
                    console.log('Port disconnected, cleaning up...');
                });

                port.onMessage.addListener((msg, port) => {
                    const handleMessage = async () => {
                        try {
                            if (msg.type === 'GET_INITIAL_STATE') {
                                await this.sendPopupUpdate(port);
                            }
                        } catch (error) {
                            console.error('Message handler error:', error);
                        }
                    };
                    handleMessage();
                });

                // Initial update with retry mechanism
                const attemptInitialUpdate = async (retryCount = 0) => {
                    try {
                        if (!this.isValidPort(port)) return;
                        await this.checkApiStatus(true); // Force fresh check
                        await this.sendPopupUpdate(port);
                    } catch (error) {
                        if (retryCount < this.maxPortRetries) {
                            setTimeout(() => attemptInitialUpdate(retryCount + 1), 1000);
                        }
                    }
                };
                attemptInitialUpdate();
            }
        });

        // Existing message handlers
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            if (request.type === 'UPDATE_STATS') {
                this.stats = request.stats;
                this.updateBadge();
            }
            if (request.type === 'GET_API_STATUS') {
                // Convert Promise to callback for Chrome messaging
                this.checkApiStatus()
                    .then(status => sendResponse(status))
                    .catch(error => sendResponse({ error: error.message }));
                return true; // Keep channel open for async response
            }
            if (request.type === 'API_URL_CHANGED') {
                this.API_URL = request.apiUrl;
                this.checkApiStatus()
                    .then(() => this.startApiMonitoring())
                    .catch(console.error);
            }
        });
    }

    async sendPopupUpdate(port) {
        if (!this.isValidPort(port)) return;

        try {
            const status = await this.checkApiStatus();
            const message = {
                type: 'STATUS_UPDATE',
                data: {
                    api: status,
                    stats: this.stats,
                    timestamp: new Date().toISOString()
                }
            };

            try {
                port.postMessage(message);
            } catch (e) {
                this.popupPorts.delete(port);
                throw e;
            }
        } catch (error) {
            console.error('Error preparing popup update:', error);
            // Don't throw here to prevent cascade failures
        }
    }

    isValidPort(port) {
        if (!port) return false;
        try {
            port.postMessage({ type: 'PING' });
            return true;
        } catch {
            this.popupPorts.delete(port);
            return false;
        }
    }

    async updateAll() {
        const validPorts = Array.from(this.popupPorts).filter(port => this.isValidPort(port));
        this.popupPorts = new Set(validPorts);

        for (const port of validPorts) {
            try {
                await this.sendPopupUpdate(port);
            } catch (error) {
                console.error('Error updating port:', error);
            }
        }
    }

    async updateBadge() {
        try {
            const successRate = this.stats.analyzed > 0
                ? Math.round((this.stats.successful / this.stats.analyzed) * 100)
                : 0;
            await chrome.action.setBadgeText({ text: `${successRate}%` });
            await chrome.action.setBadgeBackgroundColor({
                color: this.getBadgeColor(successRate)
            });
        } catch (error) {
            console.error('Badge update error:', error);
        }
    }

    getBadgeColor(rate) {
        if (rate >= 90) return '#34a853'; // Green
        if (rate >= 70) return '#fbbc05'; // Yellow
        return '#ea4335'; // Red
    }

    async checkApiStatus(forceCheck = false) {
        if (!navigator.onLine) {
            return this.getOfflineStatus();
        }

        try {
            // Log the API URL being checked
            console.log('Checking API status at:', `${this.API_URL}/health`);

            const response = await fetch(`${this.API_URL}/health`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Cache-Control': 'no-cache',
                },
                mode: 'cors',
                cache: 'no-cache'
            });

            console.log('API Raw Response:', response);

            const data = await response.json();
            console.log('API Response Data:', data);

            // Explicitly check the response status and data
            const isAvailable = response.ok && data.status === "healthy";
            const apiStatus = this.updateApiStatus(isAvailable, {
                ...data,
                models: data.models || { vi: false, en: false }
            });

            // Broadcast the status update immediately
            await this.broadcastStatusUpdate(apiStatus);
            return apiStatus;

        } catch (error) {
            console.error('API check error:', error);
            return this.updateApiStatus(false, {
                error: error.message,
                lastError: Date.now()
            });
        }
    }

    async broadcastStatusUpdate(status) {
        // Update all content scripts
        chrome.tabs.query({}, (tabs) => {
            tabs.forEach(tab => {
                chrome.tabs.sendMessage(tab.id, {
                    type: 'API_STATUS_UPDATE',
                    status: status,
                    timestamp: new Date().toISOString()
                }).catch(() => { });
            });
        });

        // Update all popups
        this.popupPorts.forEach(port => {
            if (this.isValidPort(port)) {
                port.postMessage({
                    type: 'STATUS_UPDATE',
                    data: {
                        api: status,
                        stats: this.stats,
                        timestamp: new Date().toISOString()
                    }
                });
            }
        });
    }

    async updateStatusBadge(isAvailable, modelCount = 0) {
        try {
            const statusText = isAvailable ?
                (modelCount > 0 ? `${modelCount}` : 'ON') :
                'OFF';

            const color = isAvailable ?
                (modelCount > 0 ? '#34a853' : '#fbbc05') :
                '#ea4335';

            await Promise.all([
                chrome.action.setBadgeText({ text: statusText }),
                chrome.action.setBadgeBackgroundColor({ color })
            ]);

        } catch (error) {
            console.error('Badge update error:', error);
        }
    }

    async startApiMonitoring() {
        if (pollInterval) {
            clearInterval(pollInterval);
        }

        // Initial check
        await this.checkApiStatus();

        // Set up polling with error handling
        pollInterval = setInterval(async () => {
            try {
                const status = await this.checkApiStatus();
                if (!status.isAvailable && this.lastStatus?.isAvailable) {
                    // Status changed from available to unavailable
                    console.warn('API became unavailable');
                }
                chrome.runtime.sendMessage({
                    type: 'API_STATUS_UPDATE',
                    status: status
                }).catch(() => { });
            } catch (error) {
                console.error('API monitoring error:', error);
            }
        }, this.healthCheckInterval);
    }

    updateApiStatus(isAvailable, data = null) {
        this.lastApiCheck = Date.now();

        // Reset stats when API status changes to offline
        if (!isAvailable && this.lastStatus?.isAvailable) {
            this.stats = { analyzed: 0, successful: 0 };
        }

        const modelStatus = data?.models || { vi: false, en: false };
        const activeModels = Object.entries(modelStatus)
            .filter(([_, status]) => status)
            .map(([lang]) => lang.toUpperCase());

        this.lastStatus = {
            isAvailable,
            status: isAvailable ? 'healthy' : 'offline',
            timestamp: data?.timestamp || new Date().toISOString(),
            models: modelStatus,
            activeModels: activeModels,
            modelCount: activeModels.length,
            lastCheck: this.lastApiCheck,
            error: isAvailable ? null : (data?.error || 'Service unavailable')
        };

        // Update badge and notify
        this.updateStatusBadge(isAvailable, activeModels.length).catch(console.error);
        this.notifyStatusChange(this.lastStatus);
        this.cleanupOnStatusChange(isAvailable);

        return this.lastStatus;
    }

    async cleanupOnStatusChange(isAvailable) {
        if (!isAvailable) {
            // Reset connection attempts and clear any pending requests
            this.connectionAttempts = 0;
            this.pendingRequests.clear();

            // Clear stats and notify UI
            this.stats = { analyzed: 0, successful: 0 };
            await this.updateBadge();

            // Notify all ports of reset
            this.notifyStatusChange({
                ...this.lastStatus,
                stats: this.stats
            });
        }
    }

    async notifyStatusChange(status) {
        const validPorts = Array.from(this.popupPorts).filter(this.isValidPort);
        this.popupPorts = new Set(validPorts);

        const message = {
            type: 'STATUS_UPDATE',
            data: {
                api: status,
                stats: this.stats,
                timestamp: new Date().toISOString()
            }
        };

        validPorts.forEach(port => {
            try {
                port.postMessage(message);
            } catch (error) {
                console.warn('Failed to notify port:', error);
                this.popupPorts.delete(port);
            }
        });
    }

    getOfflineStatus() {
        return {
            isAvailable: false,
            status: 'offline',
            timestamp: new Date().toISOString(),
            models: { vi: false, en: false },
            error: 'Network offline',
            lastCheck: Date.now()
        };
    }
}

function extractComments(element) {
    const comments = [];

    if (!element) return comments;

    try {
        // Tìm tất cả role="article" elements
        const commentElements = element.querySelectorAll('[role="article"]');

        commentElements.forEach(container => {
            try {
                // Skip nếu là main post
                if (container === element) return;

                // Get author name - thử nhiều selectors
                const authorElement = (
                    container.querySelector('a[role="link"] span.x193iq5w span') ||
                    container.querySelector('a[href*="/user/"] span') ||
                    container.querySelector('a[role="link"] span')
                );
                const author = authorElement?.textContent?.trim();

                // Get comment text
                const textElement = container.querySelector('div[dir="auto"][style*="text-align"]');
                const text = textElement?.textContent?.trim();

                // Get timestamp
                const timeElement = container.querySelector('a[href*="comment_id"]');
                const time = timeElement?.textContent?.trim();

                // Lấy unique ID
                const commentId = container.getAttribute('data-commentid') ||
                    timeElement?.href?.match(/comment_id=(\d+)/)?.[1] ||
                    Date.now().toString();

                // Only add if valid
                if (author && text) {
                    comments.push({
                        id: commentId,
                        author,
                        text,
                        time,
                        isReply: isReplyComment(container)
                    });
                }
            } catch (err) {
                console.warn('Error extracting comment:', err);
            }
        });

    } catch (error) {
        console.error('Error in extractComments:', error);
    }

    return comments;
}

function isReplyComment(element) {
    return !!(
        element.closest('div[aria-label*="Reply"]') ||
        element.closest('div[aria-label*="Trả lời"]') ||
        element.closest('div[aria-label*="Phản hồi"]') ||
        element.querySelector('a[role="link"][href*="reply_comment_id"]') ||
        element.closest('div[style*="margin-left"]') ||
        element.closest('div[style*="padding-left"]')
    );
}

// Function to check if an element is a comment section
function isCommentSection(element) {
    // Check for common comment section identifiers
    return element.querySelector('[role="article"]') !== null;
}

// Function to process mutations for comments
function processMutations(mutations) {
    mutations.forEach(mutation => {
        mutation.addedNodes.forEach(node => {
            if (node.nodeType === Node.ELEMENT_NODE) {
                if (isCommentSection(node)) {
                    const comments = extractComments(node);
                    if (comments.length > 0) {
                        // Send comments to content script
                        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
                            chrome.tabs.sendMessage(tabs[0].id, {
                                type: "NEW_COMMENTS",
                                comments: comments
                            });
                        });
                    }
                }
            }
        });
    });
}