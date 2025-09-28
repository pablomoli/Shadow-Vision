/**
 * Gesture Puppets - Main Application
 * Hand gesture recognition with Three.js 3D animations
 */

class GesturePuppetsApp {
    constructor() {
        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;

        // Current 3D models and animations
        this.currentModel = null;
        this.currentMixer = null;
        this.currentAction = null;

        // WebSocket connection
        this.ws = null;
        this.wsUrl = 'ws://localhost:8000';
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;

        // Application state
        this.isConnected = false;
        this.gestureConfig = {};
        this.currentGesture = null;
        this.confidenceThreshold = 0.8;

        // Performance tracking
        this.frameCount = 0;
        this.lastFPSTime = Date.now();
        this.fps = 0;

        // DOM elements
        this.elements = {
            threeContainer: document.getElementById('three-container'),
            cameraFeed: document.getElementById('camera-feed'),
            cameraStatus: document.getElementById('camera-status'),
            gestureIcon: document.getElementById('gesture-icon'),
            gestureName: document.getElementById('gesture-name'),
            gestureConfidence: document.getElementById('gesture-confidence'),
            confidenceFill: document.getElementById('confidence-fill'),
            loadingScreen: document.getElementById('loading-screen'),
            errorMessage: document.getElementById('error-message'),
            errorText: document.getElementById('error-text'),
            connectBtn: document.getElementById('connect-btn'),
            settingsBtn: document.getElementById('settings-btn'),
            statsBtn: document.getElementById('stats-btn')
        };

        // Gesture to emoji mapping
        this.gestureEmojis = {
            dog: '🐕',
            bird: '🐦',
            rabbit: '🐰',
            butterfly: '🦋',
            snake: '🐍'
        };

        this.init();
    }

    async init() {
        console.log('Initializing Gesture Puppets App...');

        try {
            // Initialize Three.js scene
            this.initThreeJS();

            // Set up event listeners
            this.setupEventListeners();

            // Start animation loop
            this.animate();

            // Connect to WebSocket server
            await this.connectWebSocket();

        } catch (error) {
            console.error('App initialization failed:', error);
            this.showError('Initialization failed: ' + error.message);
        }
    }

    initThreeJS() {
        console.log('Initializing Three.js...');

        // Create scene
        this.scene = new THREE.Scene();

        // Create camera
        const aspectRatio = window.innerWidth / window.innerHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.1, 1000);
        this.camera.position.set(0, 2, 5);

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000000, 0);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // Add renderer to DOM
        this.elements.threeContainer.appendChild(this.renderer.domElement);

        // Add controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxDistance = 20;
        this.controls.minDistance = 2;

        // Set up lighting
        this.setupLighting();

        // Create default scene
        this.createDefaultScene();

        console.log('Three.js initialized successfully');
    }

    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);

        // Directional light (sun)
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);

        // Point lights for ambiance
        const pointLight1 = new THREE.PointLight(0x4FC3F7, 0.5, 20);
        pointLight1.position.set(-5, 5, 5);
        this.scene.add(pointLight1);

        const pointLight2 = new THREE.PointLight(0xFF6B9D, 0.3, 15);
        pointLight2.position.set(5, 3, -5);
        this.scene.add(pointLight2);
    }

    createDefaultScene() {
        // Create ground plane
        const groundGeometry = new THREE.PlaneGeometry(20, 20);
        const groundMaterial = new THREE.MeshLambertMaterial({
            color: 0x2C3E50,
            transparent: true,
            opacity: 0.8
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Add some basic geometric shapes as placeholders
        this.createPlaceholderModels();

        // Set default background
        this.setSceneBackground('default');
    }

    createPlaceholderModels() {
        // Create a simple cube as placeholder
        const geometry = new THREE.BoxGeometry(1, 1, 1);
        const material = new THREE.MeshPhongMaterial({
            color: 0x4FC3F7,
            transparent: true,
            opacity: 0.8
        });

        const cube = new THREE.Mesh(geometry, material);
        cube.position.set(0, 1, 0);
        cube.castShadow = true;
        this.scene.add(cube);

        // Add floating animation
        this.animatePlaceholder(cube);
    }

    animatePlaceholder(object) {
        const startY = object.position.y;

        const animate = () => {
            if (object.parent) {
                object.position.y = startY + Math.sin(Date.now() * 0.002) * 0.3;
                object.rotation.y += 0.01;
                requestAnimationFrame(animate);
            }
        };

        animate();
    }

    setSceneBackground(type) {
        const colors = {
            'park': 0x87CEEB,      // Sky blue
            'sky': 0x4169E1,       // Royal blue
            'meadow': 0x90EE90,    // Light green
            'garden': 0xFFB6C1,    // Light pink
            'desert': 0xDEB887,    // Burlywood
            'default': 0x2C3E50    // Dark blue-gray
        };

        const color = colors[type] || colors.default;
        this.scene.fog = new THREE.Fog(color, 10, 50);

        // Create gradient background
        this.createGradientBackground(color);
    }

    createGradientBackground(baseColor) {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 256;

        const ctx = canvas.getContext('2d');
        const gradient = ctx.createLinearGradient(0, 0, 0, 256);

        const color1 = new THREE.Color(baseColor);
        const color2 = new THREE.Color(baseColor).multiplyScalar(0.3);

        gradient.addColorStop(0, `rgb(${Math.floor(color1.r * 255)}, ${Math.floor(color1.g * 255)}, ${Math.floor(color1.b * 255)})`);
        gradient.addColorStop(1, `rgb(${Math.floor(color2.r * 255)}, ${Math.floor(color2.g * 255)}, ${Math.floor(color2.b * 255)})`);

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 256, 256);

        const texture = new THREE.CanvasTexture(canvas);
        this.scene.background = texture;
    }

    setupEventListeners() {
        // Window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Control buttons
        this.elements.connectBtn.addEventListener('click', () => this.toggleConnection());
        this.elements.settingsBtn.addEventListener('click', () => this.showSettings());
        this.elements.statsBtn.addEventListener('click', () => this.showStats());

        // Keyboard controls
        document.addEventListener('keydown', (event) => this.onKeyDown(event));
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    onKeyDown(event) {
        switch(event.code) {
            case 'Space':
                event.preventDefault();
                this.toggleConnection();
                break;
            case 'KeyR':
                this.resetCamera();
                break;
            case 'KeyF':
                this.toggleFullscreen();
                break;
        }
    }

    async connectWebSocket() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            return;
        }

        console.log('Connecting to WebSocket server...');
        this.updateConnectionStatus('Connecting...');

        try {
            this.ws = new WebSocket(this.wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('Connected');
                this.elements.connectBtn.textContent = 'Disconnect';
                this.hideLoading();
            };

            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(JSON.parse(event.data));
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus('Disconnected');
                this.elements.connectBtn.textContent = 'Connect';
                this.scheduleReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showError('Connection failed. Please check if the server is running.');
            };

        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.showError('Failed to connect to server: ' + error.message);
        }
    }

    handleWebSocketMessage(data) {
        switch(data.type) {
            case 'config':
                this.handleConfig(data.data);
                break;
            case 'frame':
                this.handleFrame(data.data);
                break;
            case 'statistics':
                this.handleStatistics(data.data);
                break;
            case 'pong':
                console.log('Pong received');
                break;
        }
    }

    handleConfig(config) {
        console.log('Received configuration:', config);
        this.gestureConfig = config;
        this.confidenceThreshold = config.confidence_threshold || 0.8;

        if (!config.has_model) {
            this.showWarning('No trained model available. Running in demo mode.');
        }
    }

    handleFrame(frameData) {
        // Update camera feed
        if (frameData.frame) {
            this.elements.cameraFeed.src = frameData.frame;
        }

        // Update camera statistics
        if (frameData.camera_stats) {
            const stats = frameData.camera_stats;
            this.elements.cameraStatus.textContent =
                `${stats.current_fps} FPS | ${stats.frames_captured} frames`;
        }

        // Handle gesture detection
        if (frameData.gesture) {
            this.handleGestureDetection(frameData.gesture);
        }
    }

    async handleGestureDetection(gestureData) {
        const { gesture, confidence, animation, scene_config } = gestureData;

        console.log('Gesture detected:', gesture, 'Confidence:', confidence);

        // Update UI
        this.updateGestureDisplay(gesture, confidence);

        // Only trigger animation if confidence is high enough
        if (confidence >= this.confidenceThreshold) {
            await this.triggerGestureAnimation(gesture, animation, scene_config);
        }
    }

    updateGestureDisplay(gesture, confidence) {
        // Update gesture icon
        const emoji = this.gestureEmojis[gesture] || '🤖';
        this.elements.gestureIcon.textContent = emoji;

        // Update gesture name
        this.elements.gestureName.textContent = gesture || 'Unknown';

        // Update confidence
        const confidencePercent = Math.round(confidence * 100);
        this.elements.gestureConfidence.textContent = `${confidencePercent}% confidence`;

        // Update confidence bar
        this.elements.confidenceFill.style.width = `${confidencePercent}%`;

        // Add pulse animation for high confidence
        if (confidence >= this.confidenceThreshold) {
            this.elements.gestureIcon.classList.add('pulse');
            setTimeout(() => {
                this.elements.gestureIcon.classList.remove('pulse');
            }, 2000);
        }
    }

    async triggerGestureAnimation(gesture, animation, sceneConfig) {
        try {
            // Don't repeat the same gesture too quickly
            if (this.currentGesture === gesture && Date.now() - this.lastGestureTime < 3000) {
                return;
            }

            this.currentGesture = gesture;
            this.lastGestureTime = Date.now();

            console.log('Triggering animation:', gesture, animation);

            // Change scene background
            if (sceneConfig && sceneConfig.scene) {
                this.setSceneBackground(sceneConfig.scene);
            }

            // For now, create a simple animated object since we don't have 3D models
            await this.createGestureVisualization(gesture, sceneConfig);

        } catch (error) {
            console.error('Animation trigger failed:', error);
        }
    }

    async createGestureVisualization(gesture, config) {
        // Remove previous model
        if (this.currentModel) {
            this.scene.remove(this.currentModel);
        }

        // Create visualization based on gesture
        const visualizations = {
            dog: () => this.createDogVisualization(),
            bird: () => this.createBirdVisualization(),
            rabbit: () => this.createRabbitVisualization(),
            butterfly: () => this.createButterflyVisualization(),
            snake: () => this.createSnakeVisualization()
        };

        const createVisualization = visualizations[gesture] || visualizations.dog;
        this.currentModel = createVisualization();

        if (this.currentModel) {
            this.scene.add(this.currentModel);
            this.animateModel(this.currentModel, gesture);
        }
    }

    createDogVisualization() {
        const group = new THREE.Group();

        // Body
        const bodyGeometry = new THREE.BoxGeometry(1.5, 0.8, 0.8);
        const bodyMaterial = new THREE.MeshPhongMaterial({ color: 0x8B4513 });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.set(0, 0.5, 0);
        group.add(body);

        // Head
        const headGeometry = new THREE.SphereGeometry(0.4);
        const head = new THREE.Mesh(headGeometry, bodyMaterial);
        head.position.set(0.7, 0.7, 0);
        group.add(head);

        // Ears
        const earGeometry = new THREE.ConeGeometry(0.15, 0.3);
        const leftEar = new THREE.Mesh(earGeometry, bodyMaterial);
        leftEar.position.set(0.6, 1.0, 0.2);
        const rightEar = new THREE.Mesh(earGeometry, bodyMaterial);
        rightEar.position.set(0.6, 1.0, -0.2);
        group.add(leftEar, rightEar);

        // Tail
        const tailGeometry = new THREE.CylinderGeometry(0.05, 0.1, 0.5);
        const tail = new THREE.Mesh(tailGeometry, bodyMaterial);
        tail.position.set(-0.7, 0.8, 0);
        tail.rotation.z = Math.PI / 4;
        group.add(tail);

        return group;
    }

    createBirdVisualization() {
        const group = new THREE.Group();

        // Body
        const bodyGeometry = new THREE.EllipsoidGeometry ?
            new THREE.EllipsoidGeometry(0.3, 0.5, 0.2) :
            new THREE.SphereGeometry(0.4);
        const bodyMaterial = new THREE.MeshPhongMaterial({ color: 0x4169E1 });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.set(0, 1, 0);
        group.add(body);

        // Wings
        const wingGeometry = new THREE.BoxGeometry(0.8, 0.1, 0.4);
        const wingMaterial = new THREE.MeshPhongMaterial({ color: 0x1E90FF });
        const leftWing = new THREE.Mesh(wingGeometry, wingMaterial);
        leftWing.position.set(-0.4, 1, 0.3);
        const rightWing = new THREE.Mesh(wingGeometry, wingMaterial);
        rightWing.position.set(-0.4, 1, -0.3);
        group.add(leftWing, rightWing);

        // Beak
        const beakGeometry = new THREE.ConeGeometry(0.05, 0.2);
        const beakMaterial = new THREE.MeshPhongMaterial({ color: 0xFFA500 });
        const beak = new THREE.Mesh(beakGeometry, beakMaterial);
        beak.position.set(0.4, 1, 0);
        beak.rotation.z = -Math.PI / 2;
        group.add(beak);

        return group;
    }

    createRabbitVisualization() {
        const group = new THREE.Group();

        // Body
        const bodyGeometry = new THREE.SphereGeometry(0.5);
        const bodyMaterial = new THREE.MeshPhongMaterial({ color: 0xFFFFFF });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.set(0, 0.5, 0);
        group.add(body);

        // Head
        const headGeometry = new THREE.SphereGeometry(0.3);
        const head = new THREE.Mesh(headGeometry, bodyMaterial);
        head.position.set(0, 1.0, 0);
        group.add(head);

        // Long ears
        const earGeometry = new THREE.EllipsoidGeometry ?
            new THREE.EllipsoidGeometry(0.1, 0.4, 0.05) :
            new THREE.CylinderGeometry(0.1, 0.05, 0.8);
        const leftEar = new THREE.Mesh(earGeometry, bodyMaterial);
        leftEar.position.set(-0.15, 1.4, 0);
        const rightEar = new THREE.Mesh(earGeometry, bodyMaterial);
        rightEar.position.set(0.15, 1.4, 0);
        group.add(leftEar, rightEar);

        return group;
    }

    createButterflyVisualization() {
        const group = new THREE.Group();

        // Body
        const bodyGeometry = new THREE.CylinderGeometry(0.02, 0.02, 0.6);
        const bodyMaterial = new THREE.MeshPhongMaterial({ color: 0x000000 });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.set(0, 1, 0);
        group.add(body);

        // Wings
        const wingGeometry = new THREE.CircleGeometry(0.3, 16);
        const wingMaterial = new THREE.MeshPhongMaterial({
            color: 0xFF69B4,
            transparent: true,
            opacity: 0.8,
            side: THREE.DoubleSide
        });

        const topLeftWing = new THREE.Mesh(wingGeometry, wingMaterial);
        topLeftWing.position.set(-0.25, 1.2, 0);
        const topRightWing = new THREE.Mesh(wingGeometry, wingMaterial);
        topRightWing.position.set(0.25, 1.2, 0);

        const bottomLeftWing = new THREE.Mesh(wingGeometry, wingMaterial);
        bottomLeftWing.position.set(-0.25, 0.8, 0);
        bottomLeftWing.scale.setScalar(0.7);
        const bottomRightWing = new THREE.Mesh(wingGeometry, wingMaterial);
        bottomRightWing.position.set(0.25, 0.8, 0);
        bottomRightWing.scale.setScalar(0.7);

        group.add(topLeftWing, topRightWing, bottomLeftWing, bottomRightWing);

        return group;
    }

    createSnakeVisualization() {
        const group = new THREE.Group();

        // Create snake body segments
        const segmentGeometry = new THREE.SphereGeometry(0.2);
        const segmentMaterial = new THREE.MeshPhongMaterial({ color: 0x228B22 });

        for (let i = 0; i < 8; i++) {
            const segment = new THREE.Mesh(segmentGeometry, segmentMaterial);
            segment.position.set(i * 0.3 - 1.2, 0.2 + Math.sin(i * 0.5) * 0.1, 0);
            group.add(segment);
        }

        return group;
    }

    animateModel(model, gestureType) {
        if (!model) return;

        const animations = {
            dog: () => this.animateDog(model),
            bird: () => this.animateBird(model),
            rabbit: () => this.animateRabbit(model),
            butterfly: () => this.animateButterfly(model),
            snake: () => this.animateSnake(model)
        };

        const animate = animations[gestureType];
        if (animate) {
            animate();
        }
    }

    animateDog(model) {
        // Tail wagging animation
        const tail = model.children.find(child =>
            child.geometry instanceof THREE.CylinderGeometry && child.position.x < 0
        );

        if (tail) {
            const originalRotation = tail.rotation.z;
            const animate = () => {
                if (model.parent) {
                    tail.rotation.z = originalRotation + Math.sin(Date.now() * 0.01) * 0.3;
                    requestAnimationFrame(animate);
                }
            };
            animate();
        }
    }

    animateBird(model) {
        // Wing flapping animation
        const wings = model.children.filter(child =>
            child.geometry instanceof THREE.BoxGeometry && child.position.x < 0
        );

        wings.forEach((wing, index) => {
            const originalY = wing.position.y;
            const animate = () => {
                if (model.parent) {
                    wing.position.y = originalY + Math.sin(Date.now() * 0.015 + index * Math.PI) * 0.1;
                    requestAnimationFrame(animate);
                }
            };
            animate();
        });
    }

    animateRabbit(model) {
        // Hopping animation
        const originalY = model.position.y;
        const animate = () => {
            if (model.parent) {
                model.position.y = originalY + Math.abs(Math.sin(Date.now() * 0.008)) * 0.3;
                requestAnimationFrame(animate);
            }
        };
        animate();
    }

    animateButterfly(model) {
        // Flutter and float animation
        const originalY = model.position.y;
        const animate = () => {
            if (model.parent) {
                model.position.y = originalY + Math.sin(Date.now() * 0.005) * 0.2;
                model.rotation.y += 0.01;

                // Wing fluttering
                model.children.forEach((wing, index) => {
                    if (wing.geometry instanceof THREE.CircleGeometry) {
                        wing.rotation.x = Math.sin(Date.now() * 0.02 + index) * 0.2;
                    }
                });

                requestAnimationFrame(animate);
            }
        };
        animate();
    }

    animateSnake(model) {
        // Slithering animation
        model.children.forEach((segment, index) => {
            const originalY = segment.position.y;
            const animate = () => {
                if (model.parent) {
                    segment.position.y = originalY + Math.sin(Date.now() * 0.01 + index * 0.5) * 0.1;
                    requestAnimationFrame(animate);
                }
            };
            animate();
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Update controls
        if (this.controls) {
            this.controls.update();
        }

        // Update any active animations
        if (this.currentMixer) {
            const delta = 0.016; // ~60fps
            this.currentMixer.update(delta);
        }

        // Render scene
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }

        // Update FPS counter
        this.updateFPS();
    }

    updateFPS() {
        this.frameCount++;
        const now = Date.now();

        if (now - this.lastFPSTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastFPSTime = now;
        }
    }

    // UI Methods
    updateConnectionStatus(status) {
        this.elements.cameraStatus.textContent = status;
    }

    hideLoading() {
        this.elements.loadingScreen.classList.add('hidden');
    }

    showError(message) {
        this.elements.errorText.textContent = message;
        this.elements.errorMessage.classList.remove('hidden');
        this.elements.loadingScreen.classList.add('hidden');
    }

    showWarning(message) {
        console.warn(message);
        // Could add a warning UI element here
    }

    toggleConnection() {
        if (this.isConnected) {
            this.ws.close();
        } else {
            this.connectWebSocket();
        }
    }

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);

            console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
            setTimeout(() => this.connectWebSocket(), delay);
        } else {
            this.showError('Maximum reconnection attempts exceeded. Please refresh the page.');
        }
    }

    resetCamera() {
        this.camera.position.set(0, 2, 5);
        this.controls.reset();
    }

    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }

    showSettings() {
        // Placeholder for settings modal
        alert('Settings panel coming soon!');
    }

    showStats() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'get_statistics' }));
        }
    }

    handleStatistics(stats) {
        console.log('Server statistics:', stats);
        alert(`Server Stats:\nUptime: ${Math.round(stats.uptime_seconds)}s\nTotal Predictions: ${stats.total_predictions}\nFPS: ${this.fps}`);
    }
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.app = new GesturePuppetsApp();
});

// Export for debugging
window.GesturePuppetsApp = GesturePuppetsApp;