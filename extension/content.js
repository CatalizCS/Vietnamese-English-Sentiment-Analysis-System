// Prevent multiple initializations
if (!window.sentimentAnalyzer) {
    // Create style element first
    const createStyleSheet = () => {
        const styleSheet = document.createElement("style");
        styleSheet.id = 'sentiment-analyzer-styles';
        document.head.appendChild(styleSheet);
        return styleSheet;
    };

    class FacebookAnalyzer {
        constructor() {
            // Create styleSheet first
            this.styleSheet = document.getElementById('sentiment-analyzer-styles') || createStyleSheet();

            this.API_URL = 'http://localhost:7270';
            this.processedPosts = new Set();
            this.stats = {
                analyzed: 0,
                successful: 0
            };
            this.API_STATUS = false;
            this.MAX_RETRIES = 3;
            this.retryDelays = [1000, 2000, 4000]; // Exponential backoff
            this.init();
            this.registerMessageHandlers();
            this.initApiUrl();
            this.pendingUpdates = new Set();
            this.updateQueue = [];
            this.isProcessing = false;
            this.lastApiCheck = null;
            this.apiCheckInterval = 5000; // 5 seconds
            this.startApiStatusCheck();
            this.isConnectionReady = false;
            this.pendingMessages = [];
            this.initConnection();
            this.initStyles();
            this.readyState = false;
            this.initializeConnection();
            this.port = null;
            this.setupPort();

            // Add comment patterns for both languages
            this.COMMENT_PATTERNS = {
                en: [
                    /Comment by (.*?)$/i,
                    /Reply by (.*?)$/i,
                    /Comment from (.*?)$/i,
                    /Reply from (.*?)$/i,
                    /^(.*?)'s comment$/i,
                    /^(.*?)'s reply$/i
                ],
                vi: [
                    /Bình luận bởi (.*?)$/i,
                    /Trả lời bởi (.*?)$/i, 
                    /Phản hồi từ (.*?)$/i,
                    /Bình luận của (.*?)$/i,
                    /Trả lời của (.*?)$/i,
                    /^(.*?) đã bình luận$/i,
                    /^(.*?) đã trả lời$/i
                ]
            };
        }

        setupPort() {
            try {
                this.port = chrome.runtime.connect({ name: 'content-script' });
                
                this.port.onDisconnect.addListener(() => {
                    console.log('Port disconnected, attempting reconnect...');
                    setTimeout(() => this.setupPort(), 1000);
                });
            } catch (error) {
                console.error('Port setup error:', error);
                setTimeout(() => this.setupPort(), 1000);
            }
        }

        async initApiUrl() {
            try {
                const { apiUrl } = await chrome.storage.local.get('apiUrl');
                if (apiUrl) {
                    this.API_URL = apiUrl;
                }
            } catch (error) {
                console.error('Error loading API URL:', error);
            }
        }

        init() {
            this.observePageChanges();
            this.addInitialButtons();
            this.handleUrlChange(); // Add this line to handle URL changes
        }

        handleUrlChange() {
            let lastUrl = location.href;
            const observer = new MutationObserver(() => {
                const currentUrl = location.href;
                if (currentUrl !== lastUrl) {
                    lastUrl = currentUrl;
                    this.onUrlChanged();
                }
            });
            observer.observe(document.body, { childList: true, subtree: true });
        }

        onUrlChanged() {
            if (this.isFacebookPostUrl(location.href)) {
                this.processCurrentPost();
            }
        }

        isFacebookPostUrl(url) {
            const patterns = [
                /facebook\.com\/[^/]+\/posts\/\d+/,
                /facebook\.com\/[^/]+\/permalink\/\d+/,
                /facebook\.com\/[^/]+\/photos\/[^/]+\/\d+/,
                /facebook\.com\/photo\.php\?fbid=\d+/,
                /facebook\.com\/[^/]+\/videos\/\d+/,
                /facebook\.com\/video\.php\?v=\d+/,
                /facebook\.com\/[^/]+\?story_fbid=\d+/
            ];
            return patterns.some(pattern => pattern.test(url));
        }

        processCurrentPost() {
            // Check if posts are loaded on the page
            const post = document.querySelector('[role="article"]');
            if (post) {
                // Analyze the post and its comments
                this.analyzePost(post);
            } else {
                // Wait for the post to load
                const observer = new MutationObserver((mutations, obs) => {
                    const post = document.querySelector('[role="article"]');
                    if (post) {
                        obs.disconnect();
                        this.analyzePost(post);
                    }
                });
                observer.observe(document.body, { childList: true, subtree: true });
            }
        }

        observePageChanges() {
            const observer = new MutationObserver(() => {
                this.addInitialButtons();
                this.addCommentSectionButtons();
            });
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        }

        addInitialButtons() {
            const posts = document.querySelectorAll('div[data-pagelet^="FeedUnit_"]');
            posts.forEach(post => {
                if (!this.processedPosts.has(post)) {
                    this.addAnalyzeButton(post);
                    this.processedPosts.add(post);
                }
            });
        }

        addAnalyzeButton(post) {
            const button = document.createElement('button');
            button.className = 'sentiment-analyze-btn';
            button.textContent = 'Phân tích cảm xúc';
            button.onclick = () => this.analyzePost(post);

            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'sentiment-button-container';
            buttonContainer.appendChild(button);

            const actionsBar = post.querySelector('[aria-label="Actions for this post"]');
            if (actionsBar) {
                actionsBar.parentNode.insertBefore(buttonContainer, actionsBar.nextSibling);
            } else {
                // Fallback if actions bar is not found
                post.appendChild(buttonContainer);
            }
        }

        addCommentSectionButtons() {
            // Find all comment section buttons that don't have our analyze button
            const commentButtons = document.querySelectorAll('div[role="button"]:not(.has-analyze-btn)');

            commentButtons.forEach(button => {
                // Check if it's a comment button by looking for typical text patterns
                const text = button.textContent.toLowerCase();
                if (text.includes('bình luận') || text.includes('comments')) {
                    button.classList.add('has-analyze-btn');

                    // Create analyze button
                    const analyzeBtn = document.createElement('div');
                    analyzeBtn.className = 'sentiment-analyze-btn-inline';
                    analyzeBtn.innerHTML = `
                        <div role="button" class="analyze-comments-btn">
                            <span>Phân tích</span>
                        </div>
                    `;

                    // Add click handler
                    analyzeBtn.onclick = (e) => {
                        e.stopPropagation();
                        // Find the closest article element (post container)
                        const postElement = button.closest('[role="article"]');
                        if (postElement) {
                            this.analyzePost(postElement);
                        }
                    };

                    // Insert after the comment button
                    const container = button.parentElement;
                    if (container) {
                        // Create a wrapper if needed
                        let wrapper = container.querySelector('.comment-buttons-wrapper');
                        if (!wrapper) {
                            wrapper = document.createElement('div');
                            wrapper.className = 'comment-buttons-wrapper';
                            container.appendChild(wrapper);
                        }
                        wrapper.appendChild(analyzeBtn);
                    }
                }
            });
        }

        initConnection() {
            // Send ready message and wait for acknowledgment
            chrome.runtime.sendMessage({ type: 'CONTENT_SCRIPT_READY' }, (response) => {
                if (response?.success) {
                    this.isConnectionReady = true;
                    // Process any pending messages
                    this.processPendingMessages();
                }
            });

            // Handle connection status check
            chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
                if (message.type === 'PING') {
                    sendResponse({ success: true, ready: this.isConnectionReady });
                    return true;
                }
            });
        }

        setupConnectionListener() {
            // Notify that content script is ready
            chrome.runtime.sendMessage({ type: 'CONTENT_SCRIPT_READY' }, (response) => {
                if (chrome.runtime.lastError) {
                    console.warn('Connection setup error:', chrome.runtime.lastError);
                    return;
                }
                this.isConnectionReady = true;
            });
        }

        registerMessageHandlers() {
            chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
                // Immediately send acknowledgment
                sendResponse({ received: true });

                // Handle the message asynchronously
                this.handleAsyncMessage(request).then(response => {
                    // Send response through port if available
                    if (this.port) {
                        this.port.postMessage({
                            type: 'RESPONSE',
                            requestId: request.requestId,
                            response: response
                        });
                    }
                }).catch(error => {
                    console.error('Message handling error:', error);
                    if (this.port) {
                        this.port.postMessage({
                            type: 'ERROR',
                            requestId: request.requestId,
                            error: error.message
                        });
                    }
                });

                // Return true to indicate we'll send response asynchronously
                return true;
            });
        }

        handleAsyncMessage(request) {
            return new Promise(async (resolve, reject) => {
                try {
                    if (!request || !request.type) {
                        throw new Error('Invalid message format');
                    }
    
                    let response;
                    switch (request.type) {
                        case 'ANALYZE_POST':
                            if (!request.postId) {
                                throw new Error('Missing postId parameter');
                            }
                            response = await this.analyzeFacebookPost(request.postId);
                            break;
                            
                        case 'ANALYZE_CURRENT':
                            const post = document.querySelector('[role="article"]');
                            if (!post) {
                                throw new Error('No post found on current page');
                            }
                            response = await this.analyzePost(post);
                            break;
    
                        case 'PING':
                            response = { 
                                success: true, 
                                ready: this.readyState,
                                url: window.location.href,
                                status: this.API_STATUS
                            };
                            break;
    
                        case 'GET_STATS':
                            response = { 
                                success: true, 
                                stats: this.stats,
                                analyzed: this.stats.analyzed,
                                successful: this.stats.successful
                            };
                            break;
    
                        case 'RESET_STATS':
                            this.stats = { analyzed: 0, successful: 0 };
                            response = { success: true, stats: this.stats };
                            break;
    
                        case 'UPDATE_STATE':
                            if (request.stats) {
                                Object.assign(this.stats, request.stats);
                            }
                            if (request.apiStatus !== undefined) {
                                this.API_STATUS = request.apiStatus;
                            }
                            response = { 
                                success: true, 
                                stats: this.stats,
                                apiStatus: this.API_STATUS 
                            };
                            break;
    
                        case 'API_URL_CHANGED':
                            await this.initApiUrl();
                            await this.checkApiStatus();
                            response = { 
                                success: true,
                                apiUrl: this.API_URL,
                                status: this.API_STATUS
                            };
                            break;
    
                        case 'CHECK_API':
                            const status = await this.checkApiStatus();
                            response = { 
                                success: true,
                                status: status,
                                apiUrl: this.API_URL 
                            };
                            break;
    
                        default:
                            throw new Error(`Unsupported message type: ${request.type}`);
                    }
    
                    resolve({ 
                        success: true,
                        requestId: request.requestId,
                        ...response 
                    });
    
                } catch (error) {
                    console.error('Message handling error:', error);
                    reject({ 
                        success: false, 
                        requestId: request?.requestId,
                        error: error.message || 'Unknown error occurred',
                        details: error.stack
                    });
                }
            });
        }

        async findPostElement(postId) {
            const selectors = [
                `[data-post-id="${postId}"]`,
                `[data-ft*="${postId}"]`,
                `[id*="post_content_${postId}"]`,
                `[id*="${postId}"]`,
                '[role="article"]'
            ];
            
            const element = document.querySelector(selectors.join(','));
            if (!element) {
                throw new Error('Post element not found');
            }
            return element;
        }

        async processPendingMessages() {
            while (this.pendingMessages.length > 0) {
                const { request, sender, sendResponse } = this.pendingMessages.shift();
                await this.handleMessage(request, sender, sendResponse);
            }
        }

        handleDataUpdate(data) {
            if (data.status) {
                this.API_STATUS = data.status.isAvailable;
            }
            // Process any pending updates if API is available
            if (this.API_STATUS) {
                this.processPendingUpdates();
            }
        }

        async analyzePost(post) {
            const postId = post.getAttribute('data-post-id') || Date.now().toString();

            if (this.pendingUpdates.has(postId)) {
                return; // Already processing
            }

            try {
                this.pendingUpdates.add(postId);
                const button = post.querySelector('.sentiment-analyze-btn');
                if (button) button.disabled = true;

                let totalAnalyzed = 0;
                let successfulAnalyses = 0;

                // Analyze main post content
                const postContent = post.querySelector('[data-ad-preview="message"]');
                if (postContent) {
                    const content = postContent.textContent.trim();
                    const result = await this.analyzeSentiment(content);
                    if (result) {
                        this.displayResult(postContent, result, 'Nội dung bài viết');
                        successfulAnalyses++;
                    }
                    totalAnalyzed++;
                }

                // Find and analyze comments using new method
                const comments = this.findComments(post);
                console.log(`Found ${comments.length} comments`);

                for (const comment of comments) {
                    if (!comment.text) continue;
                    
                    const loadingIndicator = this.addLoadingIndicator(comment.element);
                    
                    try {
                        const result = await this.analyzeSentiment(comment.text);
                        if (result) {
                            this.displayResult(
                                comment.element, 
                                result,
                                `Bình luận của ${comment.userName}`
                            );
                            successfulAnalyses++;
                        }
                        totalAnalyzed++;
                    } finally {
                        loadingIndicator.remove();
                    }
                }

                // Update statistics
                this.stats.analyzed += totalAnalyzed;
                this.stats.successful += successfulAnalyses;
                chrome.runtime.sendMessage({
                    type: 'UPDATE_STATS',
                    stats: this.stats
                });

                if (button) {
                    button.disabled = false;
                    button.textContent = 'Phân tích lại';
                }

            } catch (error) {
                console.error('Analysis error:', error);
                this.showError('Có lỗi xảy ra khi phân tích');
            } finally {
                this.pendingUpdates.delete(postId);
            }
        }

        async checkApiStatus() {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 3000);

                const response = await fetch(`${this.API_URL}/health`, {
                    signal: controller.signal,
                    method: 'GET',
                    headers: {
                        'Accept': 'application/json',
                        'Cache-Control': 'no-cache'
                    },
                    mode: 'cors',
                    cache: 'no-cache'
                });

                clearTimeout(timeoutId);

                if (response.ok) {
                    const data = await response.json();
                    const models = data.models || { vi: false, en: false };
                    const activeModels = Object.values(models).filter(status => status).length;

                    this.API_STATUS = data.status === "healthy" && activeModels > 0;
                    return this.API_STATUS;
                }

                this.API_STATUS = false;
                return false;

            } catch (error) {
                console.error('API check error:', error);
                this.API_STATUS = false;
                return false;
            }
        }

        async analyzeSentiment(text, retryCount = 0) {
            if (!this.API_STATUS && !(await this.checkApiStatus())) {
                this.showError('API không khả dụng. Vui lòng thử lại sau.');
                return null;
            }

            try {
                console.log('Sending request to:', `${this.API_URL}/predict`);

                const response = await fetch(`${this.API_URL}/predict`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json',
                    },
                    mode: 'cors',
                    cache: 'no-cache',
                    body: JSON.stringify({
                        text: text,
                        language: 'vi'
                    })
                });

                console.log('API Response:', response.status, response.statusText);

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                console.log('API Result:', result);
                return result;
            } catch (error) {
                if (retryCount < this.MAX_RETRIES) {
                    await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1)));
                    return this.analyzeSentiment(text, retryCount + 1);
                }
                this.API_STATUS = false;
                console.error('API error:', error);
                return null;
            }
        }

        async processPendingUpdates() {
            if (this.isProcessing || this.updateQueue.length === 0) return;

            this.isProcessing = true;
            while (this.updateQueue.length > 0) {
                const update = this.updateQueue.shift();
                try {
                    await this.processUpdate(update);
                } catch (error) {
                    console.error('Update processing error:', error);
                }
            }
            this.isProcessing = false;
        }

        async processUpdate(update) {
            try {
                switch (update.type) {
                    case 'POST_ANALYSIS':
                        await this.analyzePost(update.post);
                        break;
                    case 'COMMENT_ANALYSIS':
                        await this.analyzeComment(update.comment);
                        break;
                    case 'BATCH_ANALYSIS':
                        await this.analyzeBatch(update.items);
                        break;
                    case 'UPDATE_UI':
                        await this.updateUI(update.data);
                        break;
                    case 'REFRESH_BUTTONS':
                        this.addInitialButtons();
                        this.addCommentSectionButtons();
                        break;
                    case 'UPDATE_STYLES':
                        this.updateStyles(update.styles);
                        break;
                    default:
                        console.warn('Unknown update type:', update.type);
                }
            } catch (error) {
                console.error('Error processing update:', error);
                this.updateQueue.push(update); // Re-queue failed updates
            }
        }

        async analyzeComment(comment) {
            const loadingIndicator = this.addLoadingIndicator(comment);
            try {
                const text = comment.textContent.trim();
                const result = await this.analyzeSentiment(text);
                if (result) {
                    this.displayResult(comment, result, 'Bình luận');
                    this.stats.successful++;
                }
                this.stats.analyzed++;
            } finally {
                loadingIndicator.remove();
            }
        }

        async analyzeBatch(items) {
            for (const item of items) {
                try {
                    if (item.type === 'post') {
                        await this.analyzePost(item.element);
                    } else if (item.type === 'comment') {
                        await this.analyzeComment(item.element);
                    }
                } catch (error) {
                    console.error('Batch analysis error:', error);
                }
            }
        }

        async extractPostContent(postElement) {
            try {
                const content = {
                    text: '',
                    comments: [],
                    reactions: {
                        total: 0,
                        types: {}
                    }
                };

                // Extract post content
                const postText = postElement.querySelector('[data-ad-preview="message"]');
                if (postText) {
                    content.text = postText.textContent.trim();
                }

                // Extract comments
                const commentElements = postElement.querySelectorAll('[role="article"][aria-label*="Comment"]');
                for (const comment of commentElements) {
                    try {
                        const commentData = {
                            id: comment.getAttribute('data-commentid') || Date.now().toString(),
                            text: '',
                            user: {
                                name: '',
                                profile: ''
                            },
                            timestamp: '',
                            reactions: []
                        };

                        // Get comment text
                        const commentText = comment.querySelector('[data-ad-preview="message"]');
                        if (commentText) {
                            commentData.text = commentText.textContent.trim();
                        }

                        // Get user info
                        const userLink = comment.querySelector('a[role="link"][tabindex="0"]');
                        if (userLink) {
                            commentData.user.name = userLink.textContent.trim();
                            commentData.user.profile = userLink.href;
                        }

                        // Get timestamp if available
                        const timestamp = comment.querySelector('a[role="link"][href*="comment_id"]');
                        if (timestamp) {
                            commentData.timestamp = timestamp.textContent.trim();
                        }

                        content.comments.push(commentData);
                    } catch (err) {
                        console.error('Error extracting comment:', err);
                    }
                }

                // Extract reaction counts
                const reactionBar = postElement.querySelector('[aria-label*="reaction"]');
                if (reactionBar) {
                    const reactionText = reactionBar.getAttribute('aria-label');
                    // Parse reaction counts from aria-label text
                    const counts = this.parseReactionCounts(reactionText);
                    content.reactions = counts;
                }

                return content;
            } catch (error) {
                console.error('Error extracting post content:', error);
                throw error;
            }
        }

        parseReactionCounts(text) {
            const counts = {
                total: 0,
                types: {}
            };

            try {
                // Common reaction types in Vietnamese
                const reactionTypes = {
                    'Thích': 'like',
                    'Yêu thích': 'love',
                    'Haha': 'haha',
                    'Wow': 'wow',
                    'Buồn': 'sad',
                    'Phẫn nộ': 'angry'
                };

                // Extract numbers and reaction types from text
                for (const [vn, en] of Object.entries(reactionTypes)) {
                    const regex = new RegExp(`(\\d+)\\s*${vn}`);
                    const match = text.match(regex);
                    if (match) {
                        const count = parseInt(match[1]);
                        counts.types[en] = count;
                        counts.total += count;
                    }
                }
            } catch (err) {
                console.error('Error parsing reactions:', err);
            }

            return counts;
        }

        async analyzeFacebookPost(postId) {
            try {
                const postElement = document.querySelector([
                    `div[role="article"]`,
                    `[data-post-id="${postId}"]`,
                    `[data-ft*="${postId}"]`,
                    `[id*="post_content_${postId}"]`
                ].join(','));

                if (!postElement) {
                    throw new Error('Không tìm thấy nội dung bài viết');
                }

                // Use DOM extraction method directly
                const content = await this.extractPostContent(postElement);
                
                if (!content.text && !content.comments.length) {
                    throw new Error('Không tìm thấy nội dung để phân tích');
                }

                let totalAnalyzed = 0;
                let successfulAnalyses = 0;

                // Analyze main post content
                if (content.text) {
                    const postResult = await this.analyzeSentiment(content.text);
                    if (postResult) {
                        this.displayResult(
                            postElement.querySelector('[data-ad-preview="message"]'),
                            postResult,
                            'Nội dung bài viết'
                        );
                        successfulAnalyses++;
                    }
                    totalAnalyzed++;
                }

                // Find and analyze comments
                const comments = this.findComments(postElement);
                for (const comment of comments) {
                    if (!comment.text) continue;
                    
                    const loadingIndicator = this.addLoadingIndicator(comment.element);
                    try {
                        const result = await this.analyzeSentiment(comment.text);
                        if (result) {
                            this.displayResult(
                                comment.element,
                                result,
                                `Bình luận của ${comment.userName}`
                            );
                            successfulAnalyses++;
                        }
                        totalAnalyzed++;
                    } finally {
                        loadingIndicator.remove();
                    }
                }

                this.stats.analyzed += totalAnalyzed;
                this.stats.successful += successfulAnalyses;

                return {
                    success: true,
                    analyzed: totalAnalyzed,
                    successful: successfulAnalyses
                };

            } catch (error) {
                console.error('Analysis error:', error);
                this.showError(error.message || 'Có lỗi xảy ra khi phân tích');
                throw error;
            }
        }

        displayResult(element, result, label = '') {
            if (!result) return;

            const resultDiv = document.createElement('div');
            resultDiv.className = `sentiment-result sentiment-${this.getSentimentClass(result.sentiment)}`;

            resultDiv.innerHTML = `
                ${label ? `<strong>${label}:</strong><br>` : ''}
                ${result.emotion.emoji} 
                <strong>${result.sentiment_label}</strong> - 
                ${result.emotion.label}<br>
                Độ tin cậy: ${(result.confidence * 100).toFixed(1)}%
            `;

            element.parentNode.insertBefore(resultDiv, element.nextSibling);
        }

        getSentimentClass(sentiment) {
            return {
                2: 'positive',
                1: 'neutral',
                0: 'negative'
            }[sentiment] || 'neutral';
        }

        addLoadingIndicator(element) {
            const indicator = document.createElement('div');
            indicator.className = 'sentiment-loading';
            indicator.innerHTML = '<div class="spinner"></div>';
            element.parentNode.insertBefore(indicator, element.nextSibling);
            return indicator;
        }

        showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'sentiment-error';
            errorDiv.textContent = message;
            document.body.appendChild(errorDiv);
            setTimeout(() => errorDiv.remove(), 3000);
        }

        startApiStatusCheck() {
            // Check API status periodically
            setInterval(async () => {
                await this.checkApiStatus();
            }, this.apiCheckInterval);
        }

        initStyles() {
            const styles = `
                .comment-buttons-wrapper {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                
                .sentiment-analyze-btn-inline {
                    display: inline-flex;
                    align-items: center;
                    cursor: pointer;
                    color: #65676B;
                    font-size: inherit;
                    padding: 4px 8px;
                    border-radius: 6px;
                    transition: background-color 0.2s;
                }
                
                .sentiment-analyze-btn-inline:hover {
                    background-color: rgba(0, 0, 0, 0.05);
                }
                
                .analyze-comments-btn {
                    display: flex;
                    align-items: center;
                    gap: 4px;
                }
                
                .analyze-comments-btn svg {
                    width: 16px;
                    height: 16px;
                    fill: currentColor;
                }
                
                .sentiment-loading-overlay {
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(255, 255, 255, 0.8);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 1000;
                }
                
                .sentiment-stats {
                    margin-top: 8px;
                    font-size: 12px;
                    color: #65676B;
                }
                
                .sentiment-error {
                    animation: fadeOut 0.3s ease-in-out forwards;
                    animation-delay: 2.7s;
                }
                
                @keyframes fadeOut {
                    from { opacity: 1; }
                    to { opacity: 0; }
                }
            `;

            this.styleSheet.textContent += styles;
        }

        async initializeConnection() {
            try {
                // Signal that content script is ready
                chrome.runtime.sendMessage({ 
                    type: 'CONTENT_SCRIPT_READY', 
                    url: window.location.href 
                }, (response) => {
                    if (chrome.runtime.lastError) {
                        console.warn('Initial connection failed:', chrome.runtime.lastError);
                        // Retry after delay
                        setTimeout(() => this.initializeConnection(), 1000);
                        return;
                    }
                    this.readyState = true;
                    this.processPendingMessages();
                });

                // Handle connection requests
                chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
                    if (message.type === 'PING') {
                        sendResponse({ 
                            success: true, 
                            ready: this.readyState,
                            url: window.location.href 
                        });
                        return true;
                    }
                    return false;
                });

            } catch (error) {
                console.error('Connection initialization error:', error);
                // Retry after delay
                setTimeout(() => this.initializeConnection(), 1000);
            }
        }

        findComments(postElement) {
            const comments = [];
            const elements = Array.from(postElement.getElementsByTagName('*'));
            
            for (const element of elements) {
                const ariaLabel = element.getAttribute('aria-label') || '';
                let isComment = false;
                let userName = '';

                // Check English patterns
                for (const pattern of this.COMMENT_PATTERNS.en) {
                    const match = ariaLabel.match(pattern);
                    if (match) {
                        isComment = true;
                        userName = match[1];
                        break;
                    }
                }

                // Check Vietnamese patterns if not found in English
                if (!isComment) {
                    for (const pattern of this.COMMENT_PATTERNS.vi) {
                        const match = ariaLabel.match(pattern);
                        if (match) {
                            isComment = true;
                            userName = match[1];
                            break;
                        }
                    }
                }

                if (isComment && element.getAttribute('role') === 'article') {
                    comments.push({
                        element: element,
                        userName: userName,
                        id: element.getAttribute('data-commentid') || Date.now().toString(),
                        text: this.extractCommentText(element)
                    });
                }
            }

            return comments;
        }

        extractCommentText(commentElement) {
            // Find text content using common patterns
            const contentPatterns = [
                '[data-ad-preview="message"]',
                '[data-ad-comet-preview="message"]',
                '.userContent',
                '.UFICommentBody'
            ];

            for (const pattern of contentPatterns) {
                const element = commentElement.querySelector(pattern);
                if (element) {
                    return element.textContent.trim();
                }
            }

            // Fallback: Try to find text content in any paragraph elements
            const paragraphs = commentElement.getElementsByTagName('p');
            if (paragraphs.length > 0) {
                return Array.from(paragraphs)
                    .map(p => p.textContent.trim())
                    .filter(text => text)
                    .join(' ');
            }

            return '';
        }
    }

    // Create single instance
    window.sentimentAnalyzer = new FacebookAnalyzer();

    // Initialize styles only once
    if (!document.getElementById('sentiment-analyzer-styles')) {
        const styleSheet = document.createElement("style");
        styleSheet.id = 'sentiment-analyzer-styles';
        styleSheet.textContent = `
            .sentiment-analyze-btn {
                background: #1877f2;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                border: none;
                font-weight: 500;
                cursor: pointer;
                transition: background 0.2s;
            }

            .sentiment-analyze-btn:hover {
                background: #166fe5;
            }

            .sentiment-analyze-btn:disabled {
                background: #8ab4f8;
                cursor: not-allowed;
            }

            .sentiment-result {
                margin: 8px 0;
                padding: 8px 12px;
                border-radius: 8px;
                font-size: 13px;
                line-height: 1.4;
            }

            .sentiment-positive {
                background: #e6f4ea;
                border: 1px solid #34a853;
            }

            .sentiment-negative {
                background: #fce8e6;
                border: 1px solid #ea4335;
            }

            .sentiment-neutral {
                background: #f1f3f4;
                border: 1px solid #5f6368;
            }

            .sentiment-loading {
                display: flex;
                justify-content: center;
                margin: 8px 0;
            }

            .spinner {
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #1877f2;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            .sentiment-error {
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: #fce8e6;
                color: #ea4335;
                padding: 12px 20px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                z-index: 9999;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(styleSheet);
    }
}