let port;
let reconnectTimeout = null;
let updateInterval = null;
let lastUpdateTime = null;
const UPDATE_THRESHOLD = 2000; // 2 seconds

function updatePopupUI(data) {
    try {
        // Required elements lookup with fallbacks
        const elements = {
            statusIndicator: document.getElementById('status-indicator') || createElementIfMissing('status-indicator'),
            statusBadge: document.getElementById('status-badge') || createElementIfMissing('status-badge'),
            apiStatus: document.getElementById('apiStatus') || createElementIfMissing('apiStatus'),
            analyzedCount: document.getElementById('analyzed-count') || createElementIfMissing('analyzed-count'),
            successRate: document.getElementById('success-rate') || createElementIfMissing('success-rate'),
            lastUpdate: document.getElementById('last-update') || createElementIfMissing('last-update'),
            errorContainer: document.getElementById('error-container') || createElementIfMissing('error-container')
        };

        // Show error if any elements are missing
        const missingElements = Object.entries(elements)
            .filter(([key, element]) => !element)
            .map(([key]) => key);

        if (missingElements.length > 0) {
            showError(`Missing UI elements: ${missingElements.join(', ')}`);
            return;
        }

        // Hide error container if everything is OK
        elements.errorContainer.classList.add('hidden');

        // Rest of existing updatePopupUI code...
        if (!data || !data.api) return;

        // Update status indicator
        if (data.api.isAvailable) {
            elements.statusIndicator.className = 'h-3 w-3 rounded-full bg-green-500';
            elements.statusBadge.textContent = 'Online';
            elements.statusBadge.className = 'text-xs font-medium bg-green-100 text-green-800 px-2 py-1 rounded-full';
            
            if (data.api.models) {
                const modelStatus = Object.entries(data.api.models)
                    .map(([lang, status]) => `${lang.toUpperCase()}: ${status ? '✓' : '✗'}`)
                    .join(' | ');
                elements.apiStatus.textContent = `Connected | ${modelStatus}`;
            } else {
                elements.apiStatus.textContent = 'Connected';
            }
        } else {
            elements.statusIndicator.className = 'h-3 w-3 rounded-full bg-red-500';
            elements.statusBadge.textContent = 'Offline';
            elements.statusBadge.className = 'text-xs font-medium bg-red-100 text-red-800 px-2 py-1 rounded-full';
            elements.apiStatus.textContent = data.api.error || 'Connection failed';
        }

        // Update stats
        if (data.stats) {
            elements.analyzedCount.textContent = data.stats.analyzed || '0';
            elements.successRate.textContent = 
                `${Math.round((data.stats.successful / data.stats.analyzed) * 100) || 0}%`;
        }

        // Update timestamp
        if (data.timestamp) {
            const lastUpdate = new Date(data.timestamp);
            elements.lastUpdate.textContent = `Last updated: ${lastUpdate.toLocaleTimeString()}`;
        }
    } catch (error) {
        showError(`UI Update Error: ${error.message}`);
    }
}

function createElementIfMissing(id) {
    // Create missing element with default styling
    const element = document.createElement('div');
    element.id = id;
    element.className = 'missing-element';
    document.body.appendChild(element);
    console.warn(`Created missing element: ${id}`);
    return element;
}

function showError(message) {
    const errorContainer = document.getElementById('error-container') || 
                         createElementIfMissing('error-container');
    errorContainer.classList.remove('hidden');
    errorContainer.textContent = message;
    console.error(message);
}

function setupConnection() {
    try {
        if (port) {
            try {
                port.disconnect();
            } catch (e) {
                console.warn('Error disconnecting old port:', e);
            }
            port = null;
        }

        const connect = (retryCount = 0) => {
            try {
                port = chrome.runtime.connect({ name: 'popup' });
                setupMessageHandlers(port);
                startUpdateCycle(port);
                
                // Add auto-reconnect logic
                port.onDisconnect.addListener(() => {
                    if (chrome.runtime.lastError) {
                        console.warn('Connection lost:', chrome.runtime.lastError);
                        retryConnection(retryCount);
                    }
                });
            } catch (error) {
                console.error('Connection error:', error);
                retryConnection(retryCount);
            }
        };

        connect();
    } catch (error) {
        console.error('Setup error:', error);
    }
}

function retryConnection(retryCount) {
    port = null;
    clearInterval(updateInterval);
    if (reconnectTimeout) clearTimeout(reconnectTimeout);
    
    if (retryCount < 3) {
        reconnectTimeout = setTimeout(
            () => setupConnection(retryCount + 1), 
            1000 * Math.pow(2, retryCount)
        );
    }
}

function setupMessageHandlers(port) {
    if (!port) return;
    
    port.onMessage.addListener((msg) => {
        try {
            switch(msg.type) {
                case 'PING':
                    port.postMessage({ type: 'PONG' });
                    break;
                    
                case 'STATUS_UPDATE':
                    if (msg.data) updatePopupUI(msg.data);
                    break;
                    
                case 'API_ERROR':
                    handleApiError(msg.error);
                    break;
                    
                case 'STATS_UPDATE':
                    updateStats(msg.stats);
                    break;
                    
                default:
                    console.warn('Unknown message type:', msg.type);
            }
        } catch (error) {
            console.error('Message handler error:', error);
            showError(error.message);
        }
    });
}

function handleApiError(error) {
    const errorContainer = document.getElementById('error-container');
    if (errorContainer) {
        errorContainer.classList.remove('hidden');
        errorContainer.textContent = `API Error: ${error}`;
        setTimeout(() => {
            errorContainer.classList.add('hidden');
        }, 5000);
    }
}

function updateStats(stats) {
    const elements = {
        analyzedCount: document.getElementById('analyzed-count'),
        successRate: document.getElementById('success-rate')
    };

    if (elements.analyzedCount) {
        elements.analyzedCount.textContent = stats.analyzed || '0';
    }
    
    if (elements.successRate) {
        const rate = stats.analyzed > 0 
            ? Math.round((stats.successful / stats.analyzed) * 100) 
            : 0;
        elements.successRate.textContent = `${rate}%`;
    }
}

function startUpdateCycle(port) {
    if (updateInterval) clearInterval(updateInterval);
    
    // Initial state request
    requestUpdate(port);

    // Setup periodic updates with throttling
    updateInterval = setInterval(() => {
        const now = Date.now();
        if (!lastUpdateTime || now - lastUpdateTime >= UPDATE_THRESHOLD) {
            requestUpdate(port);
        }
    }, 3000);
}

function requestUpdate(port) {
    if (!port) return;
    
    try {
        port.postMessage({ type: 'GET_INITIAL_STATE' });
        lastUpdateTime = Date.now();
    } catch (e) {
        console.warn('Update request failed:', e);
        setupConnection();
    }
}

async function getCurrentTab() {
    try {
        const tabs = await chrome.tabs.query({
            active: true,
            currentWindow: true
        });
        
        if (!tabs || tabs.length === 0) {
            throw new Error('No active tab found');
        }
        return tabs[0];
    } catch (error) {
        console.error('Error getting current tab:', error);
        throw new Error('Could not access current tab');
    }
}

async function extractFacebookAccessToken(tab) {
    if (!tab?.id) {
        throw new Error('Invalid tab');
    }

    try {
        const result = await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: () => {
                try {
                    // Try to get EAAB token first
                    const ls = window.localStorage;
                    const tokenKeys = Object.keys(ls).filter(key => 
                        key.includes('token') || 
                        key.includes('EAAB') || 
                        key.includes('accessToken')
                    );

                    for (const key of tokenKeys) {
                        const value = ls.getItem(key);
                        if (value && value.includes('EAAB')) {
                            return value.match(/EAAB[^"]+/)[0];
                        }
                    }

                    // Fallback to user ID from cookie
                    const cookieMatch = document.cookie.match(/c_user=([^;]+)/);
                    const userId = cookieMatch ? cookieMatch[1] : null;
                    
                    if (!userId) {
                        throw new Error('Facebook access token not found');
                    }
                    
                    return userId;
                } catch (e) {
                    console.error('Error extracting token:', e);
                    return null;
                }
            }
        });

        if (!result || !result[0]?.result) {
            throw new Error('Could not extract Facebook access token');
        }

        return result[0].result;
    } catch (error) {
        console.error('Script execution error:', error);
        throw new Error('Failed to access Facebook data. Please make sure you are logged in.');
    }
}

async function analyzeFacebookPost(postId, accessToken) {
    try {
        // Get post details from Facebook API
        const postResponse = await fetch(
            `https://graph.facebook.com/v18.0/${postId}?fields=message,comments{message,id}&access_token=${accessToken}`
        );
        const postData = await postResponse.json();

        // Send all text for analysis
        const texts = [postData.message];
        if (postData.comments) {
            texts.push(...postData.comments.data.map(c => c.message));
        }

        // Send to our sentiment API
        const response = await fetch(`${this.API_URL}/batch`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                texts: texts,
                language: 'vi'
            })
        });

        return await response.json();
    } catch (error) {
        console.error('Analysis error:', error);
        throw error;
    }
}

function extractPostId(url) {
    // Try different Facebook URL patterns
    const patterns = [
        /\/posts\/(\d+)/,                    // Standard post URL
        /\/permalink\/(\d+)/,                // Permalink format
        /\?story_fbid=(\d+)/,               // Story format
        /\/photo\.php\?fbid=(\d+)/,         // Photo URL format
        /\/video\.php\?v=(\d+)/,            // Video URL format
        /\/(\d+)(?:\/)?(?:\?|$)/            // Direct ID format
    ];

    for (const pattern of patterns) {
        const match = url.match(pattern);
        if (match && match[1]) {
            return match[1];
        }
    }
    return null;
}

document.addEventListener('DOMContentLoaded', async () => {
    setupConnection();

    // Setup API configuration
    const { apiUrl } = await chrome.storage.local.get('apiUrl');
    if (apiUrl) {
        document.getElementById('apiUrl').value = apiUrl;
    }

    // API configuration save handler
    document.getElementById('saveApiConfig').addEventListener('click', async () => {
        const apiUrl = document.getElementById('apiUrl').value.trim();
        const status = document.getElementById('apiUrlStatus');
        
        if (!apiUrl) {
            status.textContent = 'Please enter an API URL';
            status.className = 'text-xs text-red-500';
            return;
        }

        try {
            await chrome.storage.local.set({ apiUrl });
            chrome.runtime.sendMessage({ type: 'API_URL_CHANGED', apiUrl });
            
            status.textContent = 'API URL saved successfully';
            status.className = 'text-xs text-green-500';
            setTimeout(() => status.textContent = '', 3000);
        } catch (error) {
            status.textContent = 'Error saving API URL';
            status.className = 'text-xs text-red-500';
        }
    });

    // Add reset handler for API configuration
    document.getElementById('resetApiConfig').addEventListener('click', async () => {
        const status = document.getElementById('apiUrlStatus');
        
        try {
            // Clear API URL from storage
            await chrome.storage.local.remove('apiUrl');
            
            // Reset input field
            document.getElementById('apiUrl').value = '';
            
            // Notify background script
            chrome.runtime.sendMessage({ 
                type: 'API_URL_CHANGED', 
                apiUrl: 'http://localhost:7270' // Reset to default
            });
            
            status.textContent = 'Configuration reset successfully';
            status.className = 'text-xs text-green-500';
            setTimeout(() => status.textContent = '', 3000);
        } catch (error) {
            status.textContent = 'Error resetting configuration';
            status.className = 'text-xs text-red-500';
        }
    });

    // Add analyze current post handler
    document.getElementById('analyzeCurrentPost').addEventListener('click', async () => {
        const button = document.getElementById('analyzeCurrentPost');
        const errorContainer = document.getElementById('error-container');
        
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Đang phân tích...';
        errorContainer.classList.add('hidden');

        try {
            // Get current tab with validation
            const tab = await getCurrentTab();
            
            if (!tab?.url?.includes('facebook.com')) {
                throw new Error('Vui lòng mở bài viết Facebook để phân tích');
            }

            // Get access token with validation
            const accessToken = await extractFacebookAccessToken(tab);

            // Extract post ID with validation
            const postId = extractPostId(tab.url);
            if (!postId) {
                throw new Error('Không tìm thấy bài viết Facebook trên trang này');
            }

            // Send message with timeout and error handling
            const response = await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Phân tích quá thời gian, vui lòng thử lại'));
                }, 30000); // 30 second timeout

                chrome.tabs.sendMessage(tab.id, {
                    type: 'ANALYZE_POST',
                    postId: postId,
                    accessToken: accessToken
                }, (response) => {
                    clearTimeout(timeout);
                    if (chrome.runtime.lastError) {
                        reject(new Error(chrome.runtime.lastError.message));
                    } else {
                        resolve(response);
                    }
                });
            });

            if (!response?.success) {
                throw new Error(response?.error || 'Phân tích thất bại');
            }

            // Hide error container on success
            errorContainer.classList.add('hidden');

        } catch (error) {
            console.error('Analysis error:', error);
            errorContainer.textContent = error.message;
            errorContainer.classList.remove('hidden');
        } finally {
            // Always reset button state
            button.disabled = false;
            button.textContent = 'Phân tích bài viết này';
        }
    });

    // ...existing code...
});

// Cleanup
window.addEventListener('unload', () => {
    if (updateInterval) clearInterval(updateInterval);
    if (reconnectTimeout) clearTimeout(reconnectTimeout);
    if (port) port.disconnect();
});

async function ensureContentScriptConnection(tab) {
    return new Promise((resolve, reject) => {
        const maxRetries = 5;
        let retryCount = 0;

        const checkConnection = () => {
            chrome.tabs.sendMessage(tab.id, { type: 'PING' }, response => {
                if (chrome.runtime.lastError || !response?.success) {
                    if (retryCount++ < maxRetries) {
                        // Check if script already exists before injecting
                        chrome.scripting.executeScript({
                            target: { tabId: tab.id },
                            func: () => Boolean(window.sentimentAnalyzer)
                        }).then(result => {
                            const exists = result[0]?.result;
                            if (!exists) {
                                // Only inject if not already present
                                chrome.scripting.executeScript({
                                    target: { tabId: tab.id },
                                    files: ['content.js']
                                }).then(() => {
                                    setTimeout(checkConnection, 500);
                                }).catch(reject);
                            } else {
                                setTimeout(checkConnection, 500);
                            }
                        }).catch(reject);
                        return;
                    }
                    reject(new Error('Could not establish connection'));
                    return;
                }
                resolve(true);
            });
        };

        checkConnection();
    });
}

async function analyzeCurrentPost(tab) {
    const button = document.getElementById('analyzeCurrentPost');
    const errorContainer = document.getElementById('error-container');
    
    try {
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Đang kết nối...';
        errorContainer.classList.add('hidden');

        // Ensure content script is ready
        await ensureContentScriptConnection(tab);
        
        // Continue with analysis
        button.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Đang phân tích...';
        
        // ...rest of the analysis code...

    } catch (error) {
        console.error('Analysis error:', error);
        errorContainer.textContent = error.message;
        errorContainer.classList.remove('hidden');
    } finally {
        button.disabled = false;
        button.textContent = 'Phân tích bài viết này';
    }
}

async function ensureContentScriptConnection(tab) {
    return new Promise((resolve, reject) => {
        const maxRetries = 5;
        let retryCount = 0;

        const checkConnection = () => {
            chrome.tabs.sendMessage(tab.id, { type: 'PING' }, response => {
                if (chrome.runtime.lastError || !response?.success) {
                    if (retryCount++ < maxRetries) {
                        // Inject content script if needed
                        chrome.scripting.executeScript({
                            target: { tabId: tab.id },
                            files: ['content.js']
                        }).then(() => {
                            setTimeout(checkConnection, 500);
                        }).catch(reject);
                        return;
                    }
                    reject(new Error('Could not establish connection'));
                    return;
                }

                if (!response.ready) {
                    if (retryCount++ < maxRetries) {
                        setTimeout(checkConnection, 500);
                        return;
                    }
                    reject(new Error('Content script not ready'));
                    return;
                }

                resolve(true);
            });
        };

        checkConnection();
    });
}

async function analyzeCurrentPost(tab) {
    const button = document.getElementById('analyzeCurrentPost');
    const errorContainer = document.getElementById('error-container');
    
    try {
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Đang kết nối...';
        
        // Ensure connection is ready
        await ensureContentScriptConnection(tab);
        
        button.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Đang phân tích...';
        
        const postId = extractPostId(tab.url);
        if (!postId) {
            throw new Error('Không tìm thấy bài viết Facebook trên trang này');
        }

        const response = await sendMessageWithRetry(tab.id, {
            type: 'ANALYZE_POST',
            postId: postId
        });

        if (!response?.success) {
            throw new Error(response?.error || 'Phân tích thất bại');
        }

        errorContainer.classList.add('hidden');

    } catch (error) {
        console.error('Analysis error:', error);
        errorContainer.textContent = error.message;
        errorContainer.classList.remove('hidden');
    } finally {
        button.disabled = false;
        button.textContent = 'Phân tích bài viết này';
    }
}

async function sendMessageWithRetry(tabId, message, maxRetries = 3) {
    const requestId = Date.now().toString();
    message.requestId = requestId;

    return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
            cleanup();
            reject(new Error('Quá thời gian chờ phản hồi'));
        }, 30000);

        const responseHandler = (response) => {
            if (response.requestId === requestId) {
                cleanup();
                if (response.success === false) {
                    reject(new Error(response.error || 'Phân tích thất bại'));
                } else {
                    resolve(response);
                }
            }
        };

        const cleanup = () => {
            clearTimeout(timeout);
            chrome.runtime.onMessage.removeListener(responseHandler);
        };

        // Listen for response
        chrome.runtime.onMessage.addListener(responseHandler);

        // Send message
        chrome.tabs.sendMessage(tabId, message, (ack) => {
            if (chrome.runtime.lastError) {
                cleanup();
                reject(new Error(chrome.runtime.lastError.message));
            }
            // Acknowledgment received, waiting for async response
        });
    });
}

async function ensureContentScriptConnection(tab) {
    let retryCount = 0;
    const maxRetries = 5;

    while (retryCount < maxRetries) {
        try {
            const response = await sendMessageWithRetry(tab.id, { type: 'PING' });
            if (response?.success) {
                return true;
            }
        } catch (error) {
            console.warn(`Connection attempt ${retryCount + 1} failed:`, error);
            
            // Inject content script if needed
            if (retryCount === 0) {
                try {
                    await chrome.scripting.executeScript({
                        target: { tabId: tab.id },
                        files: ['content.js']
                    });
                } catch (injectionError) {
                    console.error('Script injection failed:', injectionError);
                }
            }
        }
        
        retryCount++;
        if (retryCount < maxRetries) {
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }

    throw new Error('Could not establish connection to content script');
}

// Update analyze button handler
document.getElementById('analyzeCurrentPost').addEventListener('click', async () => {
    const button = document.getElementById('analyzeCurrentPost');
    const errorContainer = document.getElementById('error-container');
    
    try {
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Đang xử lý...';
        errorContainer.classList.add('hidden');

        const tab = await getCurrentTab();
        if (!tab?.url?.includes('facebook.com')) {
            throw new Error('Vui lòng mở bài viết Facebook để phân tích');
        }

        await ensureContentScriptConnection(tab);
        
        const postId = extractPostId(tab.url);
        if (!postId) {
            throw new Error('Không tìm thấy bài viết trên trang này');
        }

        const response = await sendMessageWithRetry(tab.id, {
            type: 'ANALYZE_POST',
            postId: postId
        });

        if (!response.analyzed) {
            throw new Error('Không có nội dung nào được phân tích');
        }

        // Show success message
        errorContainer.textContent = `Đã phân tích ${response.analyzed} nội dung thành công`;
        errorContainer.className = 'mt-2 p-2 rounded-md bg-green-100 text-green-700 text-sm';
        errorContainer.classList.remove('hidden');

    } catch (error) {
        console.error('Analysis error:', error);
        errorContainer.textContent = error.message || 'Có lỗi xảy ra';
        errorContainer.className = 'mt-2 p-2 rounded-md bg-red-100 text-red-700 text-sm';
        errorContainer.classList.remove('hidden');
    } finally {
        button.disabled = false;
        button.textContent = 'Phân tích bài viết này';
    }
});