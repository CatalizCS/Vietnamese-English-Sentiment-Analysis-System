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
            this.currentTheme = this.detectTheme();
            this.setupThemeObserver();

            // Thêm queue và rate limiting config
            this.analysisQueue = [];
            this.isProcessingQueue = false;
            this.BATCH_SIZE = 10; // Số lượng comments xử lý mỗi batch
            this.RATE_LIMIT_MS = 1000; // Delay giữa các lần gọi API
            this.lastAPICall = 0;
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

            if (!(post instanceof Element)) {
                console.warn('Invalid post element');
                return;
            }

            const postId = post.getAttribute('data-post-id') || Date.now().toString();

            if (this.pendingUpdates.has(postId)) {
                console.log('Post is already being analyzed');
                return;
            }

            this.pendingUpdates.add(postId);

            try {
                const button = post.querySelector('.sentiment-analyze-btn');
                if (button) {
                    button.disabled = true;
                    button.textContent = 'Đang phân tích...';
                }

                // 1. Trước tiên click nút "Xem thêm bình luận" và "Xem các trả lời"
                await this.loadMoreComments(post);

                // 2. Chờ một chút để Facebook load các bình luận
                await new Promise(resolve => setTimeout(resolve, 2000));

                // 3. Tìm tất cả bình luận bao gồm cả replies
                const comments = document.querySelectorAll('[role="article"][tabindex="-1"]');
                console.log(`Found ${comments.length} comments to analyze`);

                // 4. Phân tích từng bình luận
                let analyzed = 0;
                let successful = 0;

                const commentBatches = this.createBatches(Array.from(comments), this.BATCH_SIZE);

                for (const batch of commentBatches) {
                    await this.processBatchWithRateLimit(batch);
                }

                // 5. Cập nhật stats
                this.stats.analyzed += analyzed;
                this.stats.successful += successful;

                // 6. Gửi thông báo đến background script
                chrome.runtime.sendMessage({
                    type: 'UPDATE_STATS',
                    stats: this.stats
                });

                if (button) {
                    button.disabled = false;
                    button.textContent = 'Phân tích lại';
                }

                return {
                    success: true,
                    analyzed,
                    successful
                };

            } catch (error) {
                console.error('Error analyzing post:', error);
                this.showError('Có lỗi xảy ra khi phân tích bài viết này.');

                if (button) {
                    button.disabled = false;
                    button.textContent = 'Phân tích lại';
                }
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

            // Thêm cache key
            const cacheKey = `sentiment_${text}`;

            // Kiểm tra cache trước
            const cachedResult = await this.getFromCache(cacheKey);
            if (cachedResult) {
                return cachedResult;
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

                // Lưu kết quả vào cache
                await this.saveToCache(cacheKey, result);

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
                            this.displayResult(postContent, postResult);
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

        displayResult(element, result) {
            if (!result) return;

            const resultDiv = document.createElement('div');
            resultDiv.className = `sentiment-result sentiment-${this.getSentimentClass(result.sentiment)}`;

            resultDiv.innerHTML = `
                <div style="margin: 4px 0;">
                    <span class="emoji">${result.emotion_emoji}</span>
                    <strong>${this.getSentimentLabel(result.sentiment)}</strong>
                    - ${result.emotion_vi}
                </div>
                <span class="sentiment-confidence">
                    Độ tin cậy: ${(result.sentiment_confidence * 100).toFixed(1)}%
                </span>
            `;

            // Insert after target element with smooth animation
            resultDiv.style.opacity = '0';
            resultDiv.style.transform = 'translateY(-4px)';
            element.parentNode.insertBefore(resultDiv, element.nextSibling);

            // Trigger animation
            requestAnimationFrame(() => {
                resultDiv.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                resultDiv.style.opacity = '1';
                resultDiv.style.transform = 'translateY(0)';
            });
        }

        getSentimentLabel(sentiment) {
            return {
                2: 'Tích cực',
                1: 'Trung tính',
                0: 'Tiêu cực'
            }[sentiment] || 'Trung tính';
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

        detectTheme() {
            // Check for dark mode by looking at Facebook's background color
            const bodyBg = window.getComputedStyle(document.body).backgroundColor;
            return this.isHexColorDark(bodyBg) ? 'dark' : 'light';
        }

        isHexColorDark(color) {
            // Convert RGB/RGBA to brightness value
            const rgb = color.match(/\d+/g);
            if (!rgb || rgb.length < 3) return false;

            // Calculate relative luminance
            const brightness = (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
            return brightness < 128;
        }

        setupThemeObserver() {
            // Watch for Facebook theme changes
            const observer = new MutationObserver(() => {
                const newTheme = this.detectTheme();
                if (newTheme !== this.currentTheme) {
                    this.currentTheme = newTheme;
                    this.updateThemeStyles();
                }
            });

            observer.observe(document.body, {
                attributes: true,
                attributeFilter: ['class']
            });
        }

        updateThemeStyles() {
            const themeStyles = this.getThemeStyles();
            let styleSheet = document.getElementById('sentiment-analyzer-styles');
            if (styleSheet) {
                styleSheet.textContent = themeStyles;
            }
        }

        getThemeStyles() {
            const baseStyles = `
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
                    padding: 12px 16px;
                    border-radius: 8px;
                    font-size: 13px;
                    line-height: 1.5;
                    font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif;
                    transition: all 0.2s ease;
                }

                .sentiment-result .emoji {
                    font-size: 16px;
                    margin-right: 6px;
                    vertical-align: -2px;
                }
            `;

            const lightModeStyles = `
                .sentiment-result {
                    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
                }

                .sentiment-result strong {
                    color: #050505;
                }

                .sentiment-positive {
                    background-color: #e7f3e8;
                    border: 1px solid rgba(35, 134, 54, 0.15);
                    color: #1d4121;
                }

                .sentiment-negative {
                    background-color: #ffebe9;
                    border: 1px solid rgba(255, 129, 130, 0.15);
                    color: #67060c;
                }

                .sentiment-neutral {
                    background-color: #f0f2f5;
                    border: 1px solid rgba(0, 0, 0, 0.08);
                    color: #050505;
                }

                .sentiment-confidence {
                    background: rgba(0, 0, 0, 0.05);
                    color: #050505;
                }
            `;

            const darkModeStyles = `
                .sentiment-result {
                    box-shadow: 0 1px 2px rgba(255, 255, 255, 0.05);
                }

                .sentiment-result strong {
                    color: #e4e6eb;
                }

                .sentiment-positive {
                    background-color: rgba(45, 136, 64, 0.2);
                    border: 1px solid rgba(45, 136, 64, 0.3);
                    color: #88cf8f;
                }

                .sentiment-negative {
                    background-color: rgba(255, 69, 68, 0.2);
                    border: 1px solid rgba(255, 69, 68, 0.3);
                    color: #ff6b6b;
                }

                .sentiment-neutral {
                    background-color: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    color: #e4e6eb;
                }

                .sentiment-confidence {
                    background: rgba(255, 255, 255, 0.1);
                    color: #e4e6eb;
                }
            `;

            return `
                ${baseStyles}
                ${this.currentTheme === 'dark' ? darkModeStyles : lightModeStyles}
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            }
            `;
        }

        initStyles() {
            // Replace existing initStyles with new theme-aware version
            const styleSheet = document.getElementById('sentiment-analyzer-styles') || createStyleSheet();
            styleSheet.textContent = this.getThemeStyles();
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

        async loadMoreComments(postElement) {
            console.log('Loading all comments...');
            try {
                let continueLoading = true;
                let lastCommentCount = 0;
                let attempts = 0;
                const maxAttempts = 30; // Increase the number of attempts to load comments

                while (continueLoading && attempts < maxAttempts) {
                    const currentComments = postElement.querySelectorAll('[role="article"]').length;
                    console.log(`Current comment count: ${currentComments}`);

                    // Check if new comments are loaded
                    if (currentComments === lastCommentCount) {
                        attempts++;
                    } else {
                        attempts = 0; // Reset attempts if new comments are found
                        lastCommentCount = currentComments;
                    }

                    // Click all "view more comments" and "view previous comments" buttons
                    const moreCommentButtons = Array.from(postElement.querySelectorAll('div[role="button"]')).filter(button => {
                        const text = button.textContent.toLowerCase();
                        return (text.includes('xem thêm bình luận') ||
                            text.includes('xem các bình luận trước') ||
                            text.includes('view more comments') ||
                            text.includes('view previous comments') ||
                            text.match(/\d+\s*(bình luận|comments?)/i));
                    });

                    // Click all "view replies" buttons
                    const replyButtons = Array.from(postElement.querySelectorAll('div[role="button"]')).filter(button => {
                        const text = button.textContent.toLowerCase();
                        return (text.includes('phản hồi') ||
                            text.includes('trả lời') ||
                            text.includes('replies') ||
                            text.match(/\d+\s*(reply|repl)/i));
                    });

                    let clickedAny = false;

                    // Click "view more comments" buttons
                    for (const button of moreCommentButtons) {
                        try {
                            // Scroll to button
                            await this.smoothScrollTo(button);
                            await new Promise(r => setTimeout(r, 1000));

                            button.click();
                            clickedAny = true;
                            console.log('Clicked more comments button');
                            await new Promise(r => setTimeout(r, 2000));
                        } catch (error) {
                            console.warn('Error clicking more comments button:', error);
                        }
                    }

                    // Click "view replies" buttons
                    for (const button of replyButtons) {
                        try {
                            await this.smoothScrollTo(button);
                            await new Promise(r => setTimeout(r, 1000));

                            button.click();
                            clickedAny = true;
                            console.log('Clicked reply button');
                            await new Promise(r => setTimeout(r, 1500));
                        } catch (error) {
                            console.warn('Error clicking reply button:', error);
                        }
                    }

                    // Stop if no buttons were clicked
                    if (!clickedAny) {
                        attempts++;
                    }

                    // Scroll to the bottom to trigger lazy loading
                    await this.smoothScrollTo(postElement.lastElementChild);
                    await new Promise(r => setTimeout(r, 2000));

                    continueLoading = clickedAny || attempts < 3;
                }

                // Final count
                const finalCommentCount = postElement.querySelectorAll('[role="article"]').length - 1; // Subtract 1 for the main post
                console.log(`Finished loading comments. Total found: ${finalCommentCount}`);

            } catch (error) {
                console.error('Error loading comments:', error);
            }
        }

        // Thêm helper method để scroll mượt
        async smoothScrollTo(element) {
            if (!element) return;

            element.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });

            // Đợi scroll hoàn tất
            await new Promise(resolve => {
                let lastPos = window.scrollY;
                const checkScrollEnd = setInterval(() => {
                    if (window.scrollY === lastPos) {
                        clearInterval(checkScrollEnd);
                        resolve();
                    }
                    lastPos = window.scrollY;
                }, 50);

                // Timeout sau 3 giây nếu scroll không kết thúc
                setTimeout(() => {
                    clearInterval(checkScrollEnd);
                    resolve();
                }, 3000);
            });
        }

        findComments(postElement) {
            const comments = new Set();
            try {
                // Query all comment elements including replies
                const commentElements = postElement.querySelectorAll('[role="article"][tabindex="-1"]');

                commentElements.forEach(element => {
                    if (element === postElement) return;

                    // Extract comment data
                    const commentData = this.extractCommentData(element);
                    if (commentData) {
                        comments.add(commentData);
                    }
                });

                return Array.from(comments);
            } catch (error) {
                console.error('Error finding comments:', error);
                return Array.from(comments);
            }
        }

        extractCommentData(element) {
            try {
                // Extract comment text
                const textElement = element.querySelector('div[dir="auto"][style*="text-align"]');
                const text = textElement?.textContent?.trim();

                // Extract user info
                const userLink = element.querySelector('a[role="link"][tabindex="0"]');
                const userName = userLink?.querySelector('span.x193iq5w span')?.textContent?.trim();
                const userProfile = userLink?.href;

                // Extract timestamp 
                const timeElement = element.querySelector('a[href*="comment_id"]');
                const timestamp = timeElement?.textContent?.trim();
                const commentId = timeElement?.href?.match(/comment_id=([0-9_]+)/)?.[1];

                // Extract reaction counts
                const reactionBarText = element.querySelector('[aria-label*="reaction"]')?.getAttribute('aria-label');
                const reactions = this.parseReactionCounts(reactionBarText || '');

                // Detect if this is a reply
                const isReply = this.isReplyComment(element);
                const parentCommentId = isReply ? this.findParentCommentId(element) : null;

                // Check for top contributor badge
                const hasTopContributorBadge = !!element.querySelector('div[role="link"]')?.textContent?.includes('Top contributor');

                // Only return if we have valid text and username
                if (text && userName) {
                    return {
                        element,
                        id: commentId || `comment_${Date.now()}`,
                        text,
                        userName,
                        userProfile,
                        timestamp,
                        reactions,
                        isReply,
                        parentCommentId,
                        hasTopContributorBadge,
                        language: this.detectLanguage(text)
                    };
                }
            } catch (err) {
                console.warn('Error extracting comment data:', err);
            }
            return null;
        }

        isReplyComment(element) {
            return !!(
                element.closest('[aria-label*="Reply"], [aria-label*="Trả lời"]') ||
                element.closest('div[class*="x1nn3v0j"]') ||
                element.querySelector('a[href*="reply_comment_id"]') ||
                element.closest('div[style*="margin-left"]') ||
                element.closest('div[style*="padding-left"]')
            );
        }

        findParentCommentId(element) {
            try {
                // Try to find parent comment through DOM hierarchy
                const parentContainer = element.closest('div[class*="x1nn3v0j"]')?.parentElement;
                if (parentContainer) {
                    const parentCommentLink = parentContainer.querySelector('a[href*="comment_id"]');
                    return parentCommentLink?.href?.match(/comment_id=([0-9_]+)/)?.[1];
                }

                // Alternative: Check reply_comment_id in element's own link
                const replyLink = element.querySelector('a[href*="reply_comment_id"]');
                if (replyLink) {
                    return replyLink.href.match(/reply_comment_id=([0-9_]+)/)?.[1];
                }
            } catch (err) {
                console.warn('Error finding parent comment:', err);
            }
            return null;
        }

        detectLanguage(text) {
            // Simple language detection based on comment content patterns
            if (/[\u0041-\u005A\u0061-\u007A]/.test(text)) {
                return text.match(/[ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]/)
                    ? 'vi'
                    : 'en';
            }
            return 'vi'; // Default to Vietnamese if no clear pattern
        }

        parseReactionCounts(text) {
            const reactionCounts = {
                total: 0,
                types: {}
            };

            try {
                const patterns = {
                    like: /(\d+)\s*(like|thích)/i,
                    love: /(\d+)\s*(love|yêu thích)/i,
                    haha: /(\d+)\s*haha/i,
                    wow: /(\d+)\s*wow/i,
                    sad: /(\d+)\s*(sad|buồn)/i,
                    angry: /(\d+)\s*(angry|phẫn nộ)/i
                };

                for (const [type, pattern] of Object.entries(patterns)) {
                    const match = text.match(pattern);
                    if (match) {
                        const count = parseInt(match[1]);
                        reactionCounts.types[type] = count;
                        reactionCounts.total += count;
                    }
                }

                // If we find a single total without types
                if (reactionCounts.total === 0) {
                    const totalMatch = text.match(/(\d+)/);
                    if (totalMatch) {
                        reactionCounts.total = parseInt(totalMatch[1]);
                    }
                }
            } catch (err) {
                console.warn('Error parsing reaction counts:', err);
            }

            return reactionCounts;
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

        // Thêm các methods mới để xử lý batch và rate limiting
        createBatches(items, batchSize) {
            const batches = [];
            for (let i = 0; i < items.length; i += batchSize) {
                batches.push(items.slice(i, i + batchSize));
            }
            return batches;
        }

        async processBatchWithRateLimit(batch) {
            for (const comment of batch) {
                try {
                    const textElement = comment.querySelector('div[dir="auto"][style*="text-align"]');
                    const text = textElement?.textContent?.trim();

                    if (!text) continue;

                    // Thực hiện rate limiting
                    const now = Date.now();
                    const timeSinceLastCall = now - this.lastAPICall;
                    if (timeSinceLastCall < this.RATE_LIMIT_MS) {
                        await new Promise(resolve =>
                            setTimeout(resolve, this.RATE_LIMIT_MS - timeSinceLastCall)
                        );
                    }

                    const loadingIndicator = this.addLoadingIndicator(comment);
                    try {
                        const result = await this.analyzeSentiment(text);
                        if (result) {
                            this.displayResult(comment, result);
                            this.stats.successful++;
                        }
                        this.stats.analyzed++;
                    } finally {
                        loadingIndicator.remove();
                    }

                    this.lastAPICall = Date.now();

                } catch (err) {
                    console.warn('Error analyzing comment:', err);
                }
            }
        }

        // Thêm methods để xử lý cache
        async getFromCache(key) {
            try {
                const result = await chrome.storage.local.get(key);
                return result[key];
            } catch (error) {
                console.warn('Cache read error:', error);
                return null;
            }
        }

        async saveToCache(key, value) {
            try {
                await chrome.storage.local.set({ [key]: value });
            } catch (error) {
                console.warn('Cache write error:', error);
            }
        }

        async callSentimentAPI(text) {
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

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        }

        extractComments(element) {
            const comments = [];

            try {
                // Find all elements that match the comment criteria
                const commentElements = element.querySelectorAll('[role="article"][tabindex="-1"]');
                
                for (const container of commentElements) {
                    try {
                        // Check for comment aria-label patterns in Vietnamese and English
                        const ariaLabel = container.getAttribute('aria-label') || '';
                        
                        // Skip if not a comment element
                        if (!this.isCommentElement(ariaLabel)) continue;

                        const comment = {
                            id: container.getAttribute('data-commentid') || 
                                Date.now().toString(),
                            text: this.extractCommentText(container),
                            author: this.extractAuthorName(ariaLabel),
                            time: this.extractTimestamp(container),
                            isReply: this.isReplyComment(container)
                        };

                        if (comment.text && comment.author) {
                            comments.push(comment);
                        }
                    } catch (err) {
                        console.warn('Error extracting comment:', err);
                    }
                }
            } catch (error) {
                console.error('Error in extractComments:', error);
            }

            return comments;
        }

        isCommentElement(ariaLabel) {
            // Comment patterns for both languages
            const viPattern = /^(Bình luận|Phản hồi|Trả lời) bởi (.+)$/i;
            const enPattern = /^Comment by (.+)$/i;
            
            return viPattern.test(ariaLabel) || enPattern.test(ariaLabel);
        }

        extractAuthorName(ariaLabel) {
            // Extract name from aria-label
            const viMatch = ariaLabel.match(/^(?:Bình luận|Phản hồi|Trả lời) bởi (.+)$/i);
            const enMatch = ariaLabel.match(/^Comment by (.+)$/i);
            
            return (viMatch || enMatch)?.[1]?.trim() || null;
        }

        extractCommentText(element) {
            // Look for content div with auto direction
            const textElement = element.querySelector('div[dir="auto"][style*="text-align: start"]');
            return textElement?.textContent?.trim() || null;
        }

        extractTimestamp(element) {
            const timeLink = element.querySelector('a[href*="comment_id"]');
            return timeLink?.textContent?.trim() || null;
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
                padding: 12px 16px;
                border-radius: 8px;
                font-size: 13px;
                line-height: 1.5;
                font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
                transition: all 0.2s ease;
            }

            .sentiment-result strong {
                font-weight: 600;
                color: #050505;
            }

            .sentiment-result .emoji {
                font-size: 16px;
                margin-right: 6px;
                vertical-align: -2px;
            }

            .sentiment-positive {
                background-color: #e7f3e8;
                border: 1px solid rgba(35, 134, 54, 0.15);
                color: #1d4121;
            }

            .sentiment-negative {
                background-color: #ffebe9;
                border: 1px solid rgba(255, 129, 130, 0.15);
                color: #67060c;
            }

            .sentiment-neutral {
                background-color: #f0f2f5;
                border: 1px solid rgba(0, 0, 0, 0.08);
                color: #050505;
            }

            .sentiment-result:hover {
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                transform: translateY(-1px);
            }

            .sentiment-confidence {
                display: inline-block;
                margin-top: 4px;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 500;
                background: rgba(0, 0, 0, 0.05);
                color: inherit;
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

    // Create a set to track processed comments
    const processedComments = new Set();

    // Listen for new comments from background script
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (message.type === "NEW_COMMENTS") {
            message.comments.forEach(comment => {
                // Create unique ID for comment to avoid duplicates
                const commentId = `${comment.author}-${comment.text}-${comment.time}`;

                if (!processedComments.has(commentId)) {
                    processedComments.add(commentId);

                    // Send to API for sentiment analysis
                    analyzeSentiment(comment.text)
                        .then(sentiment => {
                            // Add sentiment result to comment
                            displaySentimentResult(sentiment, comment);
                        })
                        .catch(err => console.error('Error analyzing comment:', err));
                }
            });
        }
    });

    function displaySentimentResult(sentiment, comment) {
        // Find the comment element again using author and text
        const commentElements = document.querySelectorAll('[role="article"][tabindex="-1"]');

        for (const element of commentElements) {
            const textElement = element.querySelector('div[dir="auto"][style="text-align: start"]');
            if (textElement && textElement.textContent.includes(comment.text)) {
                // Create sentiment display element
                const sentimentDiv = document.createElement('div');
                sentimentDiv.className = `sentiment-result sentiment-${sentiment.label.toLowerCase()}`;
                sentimentDiv.innerHTML = `
                    <div style="margin: 4px 0">
                        <span class="emoji">${getSentimentEmoji(sentiment.label)}</span>
                        <strong>${sentiment.label}</strong>
                        - ${sentiment.explanation}
                    </div>
                    <span class="sentiment-confidence">
                        Độ tin cậy: ${(sentiment.score * 100).toFixed(1)}%
                    </span>
                `;

                // Insert after comment text
                textElement.appendChild(sentimentDiv);
                break;
            }
        }
    }

    // Extract comments from Facebook page
    function extractComments() {
        const comments = [];

        try {
            // Sử dụng selector chính xác hơn để lấy tất cả bình luận
            const commentElements = document.querySelectorAll('[role="article"][tabindex="-1"]');

            commentElements.forEach(element => {
                try {
                    // Skip nếu là bài post gốc
                    if (!element.closest('[aria-label*="Comment"], [aria-label*="Bình luận"]')) {
                        return;
                    }

                    // Extract text từ div chính xác 
                    const textElement = element.querySelector('[dir="auto"][style*="text-align: start"], [dir="auto"][style*="text-align"]');
                    const text = textElement?.textContent?.trim();

                    // Extract author từ link profile
                    const authorElement = element.querySelector('a[role="link"] span[class*="x1xmvt09"] span, a[role="link"] span.x193iq5w span');
                    const author = authorElement?.textContent?.trim();

                    // Extract timestamp từ liên kết comment
                    const timestampElement = element.querySelector('a[href*="comment_id"]');
                    const timestamp = timestampElement?.textContent?.trim();

                    // Extract comment ID từ href 
                    const commentId = timestampElement?.href?.match(/comment_id=(\d+)/)?.[1];

                    // Kiểm tra xem có phải là reply không
                    const isReply = this.isNestedReplyComment(element);

                    // Lấy parent comment ID nếu là reply
                    const parentCommentId = isReply ? this.getParentCommentId(element) : '';

                    // Lấy reaction count
                    const reactionElement = element.querySelector('[aria-label*="reaction"], [class*="x1bd86ct"]');
                    const reactionCount = reactionElement ? this.parseReactionCount(reactionElement) : 0;

                    // Chỉ thêm comment có text và author
                    if (text && author) {
                        comments.push({
                            id: commentId || Date.now().toString(),
                            text,
                            author,
                            timestamp,
                            isReply,
                            parentCommentId,
                            reactionCount,
                            url: window.location.href,
                            timestamp_ms: Date.now()
                        });
                    }

                } catch (err) {
                    console.error('Error parsing comment:', err);
                }
            });

        } catch (error) {
            console.error('Error extracting comments:', error);
        }

        console.log(`Found ${comments.length} comments`);
        return comments;
    }

    // Thêm helper functions mới
    function isNestedReplyComment(element) {
        return !!(
            element.closest('[aria-label*="Reply"], [aria-label*="Trả lời"]') ||
            element.closest('[style*="margin-left"]') ||
            element.closest('[style*="padding-left"]') ||
            element.closest('[class*="x1nn3v0j"], [class*="x1mh8g0r"]')?.previousElementSibling ||
            element.querySelector('a[href*="reply_comment_id"]')
        );
    }

    function getParentCommentId(element) {
        // Tìm parent comment dựa vào DOM structure
        const parentComment = element.closest('[class*="x1nn3v0j"]')
            ?.previousElementSibling
            ?.querySelector('a[href*="comment_id"]');

        return parentComment?.href.match(/comment_id=(\d+)/)?.[1] || '';
    }

    function parseReactionCount(element) {
        const text = element.getAttribute('aria-label') || element.textContent;
        const match = text.match(/\d+/);
        return match ? parseInt(match[0]) : 0;
    }

    // Monitor DOM changes for new comments
    function watchForNewComments() {
        const observer = new MutationObserver(() => {
            const comments = extractComments();
            if (comments.length > 0) {
                chrome.runtime.sendMessage({
                    type: 'NEW_COMMENTS',
                    comments: comments
                });
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    // Initialize comment extraction
    function initCommentExtraction() {
        // Extract initial comments
        const comments = extractComments();
        if (comments.length > 0) {
            chrome.runtime.sendMessage({
                type: 'INITIAL_COMMENTS',
                comments: comments
            });
        }

        // Watch for new comments
        watchForNewComments();
    }

    // Add initialization 
    document.addEventListener('DOMContentLoaded', () => {
        initCommentExtraction();
    });
}

