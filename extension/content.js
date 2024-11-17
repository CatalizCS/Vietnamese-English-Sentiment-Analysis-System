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

            // Add cache control for API status
            this.lastHealthCheck = null;
            this.healthCheckCacheTime = 30000; // Cache health check for 30 seconds
            this.healthCheckPromise = null; // Store pending health check promise
            console.log('FacebookAnalyzer initialized');
        }

        setupPort() {
            try {
                console.log('Setting up port connection');
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
                console.log('Initializing API URL');
                const { apiUrl } = await chrome.storage.local.get('apiUrl');
                if (apiUrl) {
                    this.API_URL = apiUrl;
                }
                console.log('API URL set to:', this.API_URL);
            } catch (error) {
                console.error('Error loading API URL:', error);
            }
        }

        init() {
            console.log('Initializing FacebookAnalyzer');
            this.observePageChanges();
            this.addInitialButtons();
            this.handleUrlChange(); // Add this line to handle URL changes
        }

        handleUrlChange() {
            console.log('Setting up URL change handler');
            let lastUrl = location.href;
            const observer = new MutationObserver(() => {
                const currentUrl = location.href;
                if (currentUrl !== lastUrl) {
                    lastUrl = currentUrl;
                    console.log('URL changed to:', currentUrl);
                    this.onUrlChanged();
                }
            });
            observer.observe(document.body, { childList: true, subtree: true });
        }

        onUrlChanged() {
            console.log('Handling URL change');
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
            const isMatch = patterns.some(pattern => pattern.test(url));
            console.log('isFacebookPostUrl:', isMatch);
            return isMatch;
        }

        processCurrentPost() {
            console.log('Processing current post');
            // Check if posts are loaded on the page
            const post = document.querySelector('[role="article"]');
            if (post) {
                console.log('Post found, analyzing');
                // Analyze the post and its comments
                this.analyzePost(post);
            } else {
                console.log('Post not found, setting up observer');
                // Wait for the post to load
                const observer = new MutationObserver((mutations, obs) => {
                    const post = document.querySelector('[role="article"]');
                    if (post) {
                        obs.disconnect();
                        console.log('Post loaded, analyzing');
                        this.analyzePost(post);
                    }
                });
                observer.observe(document.body, { childList: true, subtree: true });
            }
        }

        observePageChanges() {
            console.log('Observing page changes');
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
            console.log('Adding initial buttons');
            const posts = document.querySelectorAll('div[data-pagelet^="FeedUnit_"]');
            posts.forEach(post => {
                if (!this.processedPosts.has(post)) {
                    this.addAnalyzeButton(post);
                    this.processedPosts.add(post);
                }
            });
        }

        addAnalyzeButton(post) {
            console.log('Adding analyze button to post');
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
            console.log('Adding comment section buttons');
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
            console.log('Initializing connection');
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
            console.log('Setting up connection listener');
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
            console.log('Registering message handlers');
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

        async handleAsyncMessage(request) {
            console.log('Handling async message:', request);
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
                            // Find the post element first
                            const postElement = await this.findPostElement(request.postId);
                            if (!postElement) {
                                throw new Error('Post element not found');
                            }
                            response = await this.analyzeFacebookPost(postElement);
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

                        case 'API_STATUS_UPDATE':
                            this.handleApiStatusUpdate(request.status);
                            response = { success: true };
                            break;

                        case 'DATA_UPDATE':
                            this.handleDataUpdate(request.data);
                            response = { success: true };
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

        handleApiStatusUpdate(status) {
            console.log('API status updated:', status);
            this.API_STATUS = status.isAvailable;

            if (status.isAvailable) {
                this.processPendingUpdates();
            }
        }

        async findPostElement(postId) {
            console.log('Finding post element for postId:', postId);
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
            console.log('Processing pending messages');
            while (this.pendingMessages.length > 0) {
                const { request, sender, sendResponse } = this.pendingMessages.shift();
                await this.handleMessage(request, sender, sendResponse);
            }
        }

        handleDataUpdate(data) {
            console.log('Handling data update:', data);
            if (data.status) {
                this.API_STATUS = data.status.isAvailable;
            }
            // Process any pending updates if API is available
            if (this.API_STATUS) {
                this.processPendingUpdates();
            }
        }

        async analyzePost(post) {
            console.log('Analyzing post:', post);
            // Ensure 'post' is a DOM Element
            if (!(post instanceof Element)) {
                console.error('Invalid post element:', post);
                return;
            }

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
            console.log('Checking API status');
            // Return cached status if within cache time
            if (this.lastHealthCheck &&
                Date.now() - this.lastHealthCheck.timestamp < this.healthCheckCacheTime) {
                return this.lastHealthCheck.status;
            }

            // Return existing promise if health check is in progress
            if (this.healthCheckPromise) {
                return this.healthCheckPromise;
            }

            // Perform new health check
            this.healthCheckPromise = (async () => {
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

                        // Cache the result
                        this.lastHealthCheck = {
                            timestamp: Date.now(),
                            status: this.API_STATUS
                        };

                        return this.API_STATUS;
                    }

                    this.API_STATUS = false;
                    return false;

                } catch (error) {
                    console.error('API check error:', error);
                    this.API_STATUS = false;
                    return false;
                } finally {
                    this.healthCheckPromise = null;
                }
            })();

            return this.healthCheckPromise;
        }

        async analyzeSentiment(text, retryCount = 0) {
            console.log('Analyzing sentiment for text:', text);
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
            console.log('Processing pending updates');
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
            console.log('Processing update:', update);
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
            console.log('Analyzing comment:', comment);
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
            console.log('Analyzing batch of items:', items);
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
            console.log('Extracting post content from element:', postElement);
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
                const commentElements = postElement.querySelectorAll('[role="article"]');
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

        async analyzeFacebookPost(postElement) {
            console.log('Analyzing Facebook post:', postElement);
            try {
                // Validate post element
                if (!postElement || !(postElement instanceof Element)) {
                    throw new Error('Invalid post element provided');
                }

                let totalAnalyzed = 0;
                let successfulAnalyses = 0;

                // Use custom element query helper
                const postContent = this.findPostContent(postElement);
                if (postContent) {
                    const text = postContent.textContent.trim();
                    if (text) {
                        const postResult = await this.analyzeSentiment(text);
                        if (postResult) {
                            this.displayResult(postContent, postResult, 'Nội dung bài viết');
                            successfulAnalyses++;
                        }
                        totalAnalyzed++;
                    }
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

                // Update stats
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
            console.log('Starting API status check');
            // Check API status every 60 seconds instead of 5 seconds
            setInterval(async () => {
                if (!this.lastHealthCheck ||
                    Date.now() - this.lastHealthCheck.timestamp >= 60000) {
                    await this.checkApiStatus();
                }
            }, 60000); // 1 minute interval
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

        async initializeConnection(retryCount = 0) {
            console.log('Initializing connection');
            try {
                // Signal that content script is ready
                chrome.runtime.sendMessage({
                    type: 'CONTENT_SCRIPT_READY',
                    url: window.location.href
                }, (response) => {
                    if (chrome.runtime.lastError) {
                        console.warn('Initial connection failed:', chrome.runtime.lastError);
                        // Retry after delay
                        if (retryCount < this.MAX_RETRIES) {
                            setTimeout(() => this.initializeConnection(retryCount + 1), 1000);
                        }
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
                if (retryCount < this.MAX_RETRIES) {
                    setTimeout(() => this.initializeConnection(retryCount + 1), 1000);
                }
            }
        }

        /**
         * Extract comment text from a Facebook comment element
         * @param {Element} commentElement - The root comment element
         * @returns {string|null}
         */
        extractCommentText(commentElement) {
            // Tìm div có attribute dir="auto" chứa nội dung comment
            const textContainers = commentElement.querySelectorAll('div[dir="auto"]');

            for (const container of textContainers) {
                // Kiểm tra xem div này có phải là container chứa nội dung comment không
                // bằng cách verify nó không chứa các elements khác như link, button
                if (!container.querySelector('a, button') && container.textContent.trim()) {
                    return container.textContent.trim();
                }
            }

            return null;
        }

        /**
         * Extract username from a Facebook comment element
         * @param {Element} commentElement - The root comment element
         * @returns {string|null}
         */
        extractUsername(commentElement) {
            // Tìm link profile của user
            const userLinks = commentElement.querySelectorAll('a[role="link"]');

            for (const link of userLinks) {
                // Kiểm tra xem link có phải là profile link không
                const href = link.getAttribute('href');
                if (href && href.includes('/user/')) {
                    // Tìm span chứa tên người dùng
                    const spans = link.querySelectorAll('span');
                    for (const span of spans) {
                        const text = span.textContent.trim();
                        // Loại bỏ các text không phải tên người dùng như timestamp
                        if (text && !text.includes('h') && !text.match(/^\d+$/)) {
                            return text;
                        }
                    }
                }
            }

            return null;
        }

        /**
         * Extract timestamp from a Facebook comment element
         * @param {Element} commentElement - The root comment element
         * @returns {string|null}
         */
        extractTimestamp(commentElement) {
            // Tìm link chứa timestamp
            const links = commentElement.querySelectorAll('a');

            for (const link of links) {
                const text = link.textContent.trim();
                // Timestamp thường có format như "1h", "2m", "3d" etc.
                if (text.match(/^\d+[hdmsy]$/i)) {
                    return text;
                }
            }

            return null;
        }

        /**
         * Check if user is a top contributor
         * @param {Element} commentElement - The root comment element
         * @returns {boolean}
         */
        isTopContributor(commentElement) {
            const elements = commentElement.querySelectorAll('div[role="link"]');
            for (const element of elements) {
                if (element.textContent.includes('Top contributor')) {
                    return true;
                }
            }
            return false;
        }

        /**
         * Parse a single Facebook comment
         * @param {Element} commentElement - The root comment element with role="article"
         * @returns {Object} Parsed comment data
         */
        parseComment(commentElement) {
            return {
                username: this.extractUsername(commentElement),
                text: this.extractCommentText(commentElement),
                timestamp: this.extractTimestamp(commentElement),
                isTopContributor: this.isTopContributor(commentElement)
            };
        }

        /**
         * Parse all comments in a container
         * @param {Element} container - The container element
         * @returns {Array<Object>} Array of parsed comments
         */
        parseComments(container) {
            const commentElements = container.querySelectorAll('[role="article"]');
            const comments = [];

            for (const element of commentElements) {
                try {
                    const comment = this.parseComment(element);
                    if (comment.text && comment.username) { // Only add valid comments
                        comments.push(comment);
                    }
                } catch (error) {
                    console.warn('Failed to parse comment:', error);
                }
            }

            return comments;
        }

        /**
         * Utility function to observe DOM changes and parse new comments
         * @param {Element} container - The container to observe
         * @param {Function} callback - Callback function to handle new comments
         * @returns {MutationObserver}
         */
        observeNewComments(container, callback) {
            const observer = new MutationObserver((mutations) => {
                for (const mutation of mutations) {
                    const newComments = Array.from(mutation.addedNodes)
                        .filter(node => node.nodeType === 1 && node.getAttribute('role') === 'article')
                        .map(element => this.parseComment(element))
                        .filter(comment => comment.text && comment.username);

                    if (newComments.length > 0) {
                        callback(newComments);
                    }
                }
            });

            observer.observe(container, {
                childList: true,
                subtree: true
            });

            return observer;
        }

        findComments(postElement) {
            const comments = this.parseComments(postElement);
            return comments;
        }

        findPostContent(element) {
            if (!element) return null;

            // Try different selectors in order of preference
            const selectors = [
                'div[dir="auto"][style*="text-align"]',
                'div[data-ad-preview="message"]',
                'div[data-ad-comet-preview="message"]',
                // Fallback selectors
                '[role="article"] div[dir="auto"]',
                '[data-ad-preview="message"]'
            ];

            for (const selector of selectors) {
                const content = element.querySelector(selector);
                if (content?.textContent.trim()) {
                    return content;
                }
            }

            return null;
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

