// SASOK Interface JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize SASOK interface
    try {
        console.log('Initializing SASOK interface...');
        window.sasok = new SasokInterface();
        window.sasok.init();
    } catch (error) {
        console.error('Error initializing SASOK:', error);
    }
});

// Core SASOK Interface Controller
class SasokInterface {
    constructor() {
        // API endpoints
        this.API_BASE_URL = 'http://localhost:8000';
        this.ASR_SERVICE_URL = 'http://localhost:5000';
        
        // State management
        this.state = {
            recording: false,
            audioStream: null,
            mediaRecorder: null,
            audioChunks: [],
            currentEmotion: "нейтральность",
            emotionScores: {
                anger: 4,     // Злость
                sadness: 12,  // Грусть
                happiness: 53, // Радость
                fear: 8,      // Страх
                neutral: 23   // Нейтральность
            },
            economyMetrics: {
                collectiveParanoia: 34,
                emotionalInflation: 62, 
                systemStability: 78,
                painCapital: 427.8,
                painTrend: 12.4
            },
            emotionalMechanics: {
                collectiveParanoia: { active: true },
                emotionalInflation: { active: true },
                painCapital: { active: false },
                greedParadox: { active: false },
                echoControl: { active: true },
                emptinessSyndrome: { active: false },
                trustCycle: { active: false },
                illusionsEconomy: { active: false },
                toxicGenerosity: { active: false },
                adaptationRisk: { active: false },
                chaosEnergy: { active: false }
            },
            privacyMode: true,
            emotionalModeActive: true
        };
        
        // DOM elements cache
        this.elements = {};
    }
    
    // Initialize the interface
    init() {
        this.cacheElements();
        this.initEventListeners();
        this.initTabSystem();
        this.checkSystemStatus();
        console.log("SASOK интерфейс инициализирован | SASOK наблюдает, не интерпретирует");
    }
    
    // Cache DOM elements for better performance
    cacheElements() {
        try {
            console.log('Caching DOM elements...');
            // Navigation
            this.elements.navItems = document.querySelectorAll('.nav-item') || [];
            this.elements.tabContents = document.querySelectorAll('.tab-content') || [];
            
            // Control toggles
            this.elements.emotionToggle = document.getElementById('emotionToggle');
            this.elements.privacyToggle = document.getElementById('privacyToggle');
            this.elements.currentState = document.getElementById('currentState');
            
            // System status elements
            this.elements.asr_status = document.querySelector('.status-item:nth-child(2) .status');
            if (!this.elements.asr_status) {
                console.warn('ASR status element not found in DOM');
            }
            
            // AI Chat elements
            this.elements.chatHistory = document.getElementById('chatHistory');
            this.elements.userInput = document.getElementById('userInput');
            this.elements.sendBtn = document.querySelector('.send-btn');
            this.elements.voiceBtn = document.querySelector('.voice-btn');
            
            // Emotion feedback
            this.elements.emotionValues = document.querySelectorAll('.emotion-value') || [];
            
            // Economy metrics elements
            this.elements.metricValues = document.querySelectorAll('.meter-progress') || [];
            this.elements.metricLabels = document.querySelectorAll('.meter-value') || [];
            this.elements.painCapital = document.querySelector('.capital-value');
            this.elements.painTrend = document.querySelector('.capital-trend');
            
            // Economy mechanics
            this.elements.mechanicsList = document.querySelector('.mechanics-list');
            
            // Log cache results
            console.log('DOM elements cached successfully');
        } catch (error) {
            console.error('Error caching DOM elements:', error);
        }
    }
    
    // Initialize event listeners
    initEventListeners() {
        // Tab navigation
        this.elements.navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const tabId = item.getAttribute('data-tab');
                this.switchTab(tabId);
            });
        });
        
        // Control toggles
        this.elements.emotionToggle.addEventListener('change', () => {
            this.state.emotionalModeActive = this.elements.emotionToggle.checked;
            this.updateCurrentState();
        });
        
        this.elements.privacyToggle.addEventListener('change', () => {
            this.state.privacyMode = this.elements.privacyToggle.checked;
            this.updateEmotionalMetrics(); // Privacy affects metrics
        });
        
        // Chat input
        this.elements.userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
        
        this.elements.sendBtn.addEventListener('click', () => {
            this.sendMessage();
        });
        
        // Voice input with ASR integration
        this.elements.voiceBtn.addEventListener('click', () => {
            if (this.state.recording) {
                this.stopRecording();
            } else {
                this.startRecording();
            }
        });
        
        // Economic exchange button
        document.querySelector('.exchange-btn').addEventListener('click', () => {
            this.exchangePainCapital();
        });
        
        // Add window events for system status checks
        window.addEventListener('online', () => this.checkSystemStatus());
        window.addEventListener('offline', () => this.checkSystemStatus(false));
    }
    
    // Tab system 
    initTabSystem() {
        // Set first tab as active by default if not already set
        if (!document.querySelector('.tab-content.active')) {
            this.elements.tabContents[0].classList.add('active');
            this.elements.navItems[0].classList.add('active');
        }
    }
    
    // Switch tabs
    switchTab(tabId) {
        // Remove active class from all tabs and nav items
        this.elements.tabContents.forEach(tab => tab.classList.remove('active'));
        this.elements.navItems.forEach(item => item.classList.remove('active'));
        
        // Add active class to selected tab and nav item
        document.getElementById(tabId).classList.add('active');
        document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
    }
    
    // Check system status
    checkSystemStatus(online = navigator.onLine) {
        // Check ASR service
        try {
            console.log('Checking ASR service status...');
            // Используем setTimeout, чтобы предотвратить блокирование UI
            setTimeout(() => {
                fetch(`${this.ASR_SERVICE_URL}/health`, { 
                    method: 'GET',
                    // Добавим таймаут для предотвращения долгого ожидания
                    signal: AbortSignal.timeout(3000) 
                })
                .then(response => {
                    if (response.ok) {
                        this.updateServiceStatus('asr', true);
                    } else {
                        this.updateServiceStatus('asr', false);
                    }
                })
                .catch((error) => {
                    console.log('ASR service check error:', error.message);
                    this.updateServiceStatus('asr', false);
                });
            }, 500);
            
            // Update UI based on online status
            if (!online) {
                // Set all services to offline if the system is offline
                this.updateServiceStatus('asr', false);
            }
        } catch (error) {
            console.error('Error checking system status:', error);
            this.updateServiceStatus('asr', false);
        }
    }
    
    // Update service status in UI
    updateServiceStatus(service, isOnline) {
        try {
            const statusElement = this.elements.asr_status;
            
            if (!statusElement) {
                console.warn(`Status element for service '${service}' not found in DOM`);
                return;
            }
            
            if (service === 'asr') {
                if (isOnline) {
                    statusElement.classList.remove('offline');
                    statusElement.classList.add('online');
                    statusElement.textContent = 'Онлайн';
                } else {
                    statusElement.classList.remove('online');
                    statusElement.classList.add('offline');
                    statusElement.textContent = 'Оффлайн';
                }
            }
        } catch (error) {
            console.error(`Error updating service status for ${service}:`, error);
        }
    }
    
    // Update current state display
    updateCurrentState() {
        let stateText = "SASOK наблюдает, не интерпретирует";
        
        if (this.state.emotionalModeActive) {
            if (this.state.currentEmotion) {
                const emotionEmoji = this.getEmotionEmoji(this.state.currentEmotion);
                stateText = `${emotionEmoji} ${this.capitalizeFirstLetter(this.state.currentEmotion)}`;
            }
        }
        
        this.elements.currentState.textContent = stateText;
    }
    
    // Helper: Get emoji for emotion
    getEmotionEmoji(emotion) {
        const emojiMap = {
            'злость': '😡',
            'грусть': '😢',
            'радость': '😊',
            'страх': '😨',
            'нейтральность': '😐'
        };
        
        return emojiMap[emotion.toLowerCase()] || '😐';
    }
    
    // Helper: Capitalize first letter
    capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    // Send message to AI
    async sendMessage() {
        const userMessage = this.elements.userInput.value.trim();
        if (!userMessage) return;
        
        // Add user message to chat
        this.addMessageToChat(userMessage, 'user');
        
        // Clear input
        this.elements.userInput.value = '';
        
        try {
            // Send message to API
            const response = await fetch(`${this.API_BASE_URL}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: userMessage,
                    emotionalMode: this.state.emotionalModeActive,
                    privacyMode: this.state.privacyMode
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                
                // Add AI response to chat
                this.addMessageToChat(data.response, 'assistant');
                
                // Update emotional data if provided
                if (data.emotion) {
                    this.state.currentEmotion = data.emotion.dominant;
                    this.updateEmotionValues(data.emotion.scores);
                    this.updateCurrentState();
                }
                
                // Update economy metrics
                this.updateEconomyMetrics(data.economyUpdates);
            } else {
                this.addSystemMessage("Произошла ошибка при обработке сообщения.", 'error');
            }
        } catch (error) {
            console.error("Error sending message:", error);
            this.addSystemMessage("Не удалось связаться с SASOK API.", 'error');
        }
    }
    
    // Add message to chat history
    addMessageToChat(message, type) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', type);
        messageDiv.textContent = message;
        
        this.elements.chatHistory.appendChild(messageDiv);
        this.elements.chatHistory.scrollTop = this.elements.chatHistory.scrollHeight;
    }
    
    // Add system message to chat
    addSystemMessage(message, level = 'info') {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'system');
        if (level === 'error') messageDiv.classList.add('error');
        messageDiv.textContent = message;
        
        this.elements.chatHistory.appendChild(messageDiv);
        this.elements.chatHistory.scrollTop = this.elements.chatHistory.scrollHeight;
    }
    
    // Update emotion values in UI
    updateEmotionValues(scores) {
        if (!scores) return;
        
        // Update state
        this.state.emotionScores = {
            anger: Math.round(scores.anger * 100) || 0,
            sadness: Math.round(scores.sadness * 100) || 0,
            happiness: Math.round(scores.happiness * 100) || 0,
            fear: Math.round(scores.fear * 100) || 0,
            neutral: Math.round(scores.neutral * 100) || 0
        };
        
        // Update UI elements
        const emotionCells = document.querySelectorAll('.emotion-cell');
        emotionCells.forEach((cell, index) => {
            const emotionLabel = cell.querySelector('.emotion-label').textContent.toLowerCase();
            let value = 0;
            
            switch(emotionLabel) {
                case 'злость':
                    value = this.state.emotionScores.anger;
                    break;
                case 'грусть':
                    value = this.state.emotionScores.sadness;
                    break;
                case 'радость':
                    value = this.state.emotionScores.happiness;
                    break;
                case 'страх':
                    value = this.state.emotionScores.fear;
                    break;
                case 'нейтральность':
                    value = this.state.emotionScores.neutral;
                    break;
            }
            
            cell.querySelector('.emotion-value').textContent = `${value}%`;
        });
    }
    
    // Update economy metrics in UI
    updateEconomyMetrics(updates) {
        if (!updates) return;
        
        // Update state with new metrics
        Object.assign(this.state.economyMetrics, updates);
        
        // Update UI
        this.elements.metricValues.forEach((meter, index) => {
            const metricName = meter.closest('.meter').querySelector('span').textContent;
            let value = 0;
            
            if (metricName.includes('Коллективная паранойя')) {
                value = this.state.economyMetrics.collectiveParanoia;
            } else if (metricName.includes('Эмоциональная инфляция')) {
                value = this.state.economyMetrics.emotionalInflation;
            } else if (metricName.includes('Стабильность системы')) {
                value = this.state.economyMetrics.systemStability;
            }
            
            meter.style.width = `${value}%`;
            meter.setAttribute('data-value', value);
            meter.closest('.meter').querySelector('.meter-value').textContent = `${value}%`;
        });
        
        // Update pain capital
        this.elements.painCapital.textContent = this.state.economyMetrics.painCapital.toFixed(1);
        
        // Update pain trend
        const trendElement = this.elements.painTrend;
        const trendValue = this.state.economyMetrics.painTrend;
        const trendIcon = trendValue >= 0 ? '<i class="fas fa-arrow-up"></i>' : '<i class="fas fa-arrow-down"></i>';
        const trendSign = trendValue >= 0 ? '+' : '';
        
        trendElement.innerHTML = `${trendSign}${Math.abs(trendValue).toFixed(1)} ${trendIcon}`;
        trendElement.className = 'capital-trend';
        trendElement.classList.add(trendValue >= 0 ? 'positive' : 'negative');
        
        // Update mechanics active state
        this.updateMechanicsUI();
    }
    
    // Update mechanics UI
    updateMechanicsUI() {
        // This would update the mechanics list based on active state
        const mechanics = this.elements.mechanicsList.querySelectorAll('.mechanic-item');
        
        mechanics.forEach(mechanic => {
            const mechanicTitle = mechanic.querySelector('h4').textContent;
            let isActive = false;
            
            // Match the mechanic title to the state
            switch(mechanicTitle) {
                case 'Коллективная паранойя':
                    isActive = this.state.emotionalMechanics.collectiveParanoia.active;
                    break;
                case 'Эмоциональная инфляция':
                    isActive = this.state.emotionalMechanics.emotionalInflation.active;
                    break;
                case 'Капитал боли':
                    isActive = this.state.emotionalMechanics.painCapital.active;
                    break;
                case 'Парадокс жадности':
                    isActive = this.state.emotionalMechanics.greedParadox.active;
                    break;
                case 'Эхо-контроль':
                    isActive = this.state.emotionalMechanics.echoControl.active;
                    break;
                // Add other mechanics here
            }
            
            // Update UI based on active state
            mechanic.className = 'mechanic-item';
            if (isActive) {
                mechanic.classList.add('active');
                mechanic.querySelector('.mechanic-status').textContent = 'Активна';
            } else {
                mechanic.classList.add('inactive');
                mechanic.querySelector('.mechanic-status').textContent = 'Доступна';
            }
        });
    }
    
    // Exchange pain capital for resources
    exchangePainCapital() {
        if (this.state.economyMetrics.painCapital < 50) {
            this.addSystemMessage("Недостаточно капитала боли для обмена. Минимум: 50", 'error');
            return;
        }
        
        // Simulate exchange
        const exchangeRate = 0.8 - (this.state.economyMetrics.emotionalInflation / 200); // Rate decreases with inflation
        const resourcesGained = Math.floor(this.state.economyMetrics.painCapital * exchangeRate);
        
        // Update state
        this.state.economyMetrics.painCapital = 0;
        this.state.economyMetrics.painTrend = -this.state.economyMetrics.painTrend;
        
        // Update UI
        this.updateEconomyMetrics({
            painCapital: 0,
            painTrend: this.state.economyMetrics.painTrend
        });
        
        // Notify user
        this.addSystemMessage(`Боль успешно обменена. Получено ${resourcesGained} единиц ресурсов.`);
        
        // API call to record the exchange would go here
        
        // Trigger economic mechanic changes
        this.recalculateEconomyMechanics();
    }
    
    // Recalculate active economy mechanics
    recalculateEconomyMechanics() {
        // This would be a complex calculation based on many factors
        // For demo, just randomly toggle some mechanics
        this.state.emotionalMechanics.greedParadox.active = !this.state.emotionalMechanics.greedParadox.active;
        this.updateMechanicsUI();
    }
    
    // ASR Integration Functions
    
    // Start recording audio
    async startRecording() {
        if (this.state.recording) return;
        
        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.state.audioStream = stream;
            
            // Setup media recorder
            const mediaRecorder = new MediaRecorder(stream);
            this.state.mediaRecorder = mediaRecorder;
            this.state.audioChunks = [];
            
            // Handle data available event
            mediaRecorder.addEventListener('dataavailable', (event) => {
                if (event.data.size > 0) {
                    this.state.audioChunks.push(event.data);
                }
            });
            
            // Handle recording stop event
            mediaRecorder.addEventListener('stop', () => {
                this.processAudioData();
            });
            
            // Start recording
            mediaRecorder.start();
            this.state.recording = true;
            
            // Update UI to show recording status
            this.elements.voiceBtn.classList.add('recording');
            this.elements.voiceBtn.querySelector('i').className = 'fas fa-stop';
            
            // Add system message
            this.addSystemMessage("Запись голоса начата. Говорите...");
            
        } catch (error) {
            console.error("Error accessing microphone:", error);
            this.addSystemMessage("Не удалось получить доступ к микрофону.", 'error');
        }
    }
    
    // Stop recording
    stopRecording() {
        if (!this.state.recording) return;
        
        // Stop media recorder
        this.state.mediaRecorder.stop();
        
        // Stop audio tracks
        this.state.audioStream.getTracks().forEach(track => track.stop());
        
        // Update state
        this.state.recording = false;
        
        // Update UI
        this.elements.voiceBtn.classList.remove('recording');
        this.elements.voiceBtn.querySelector('i').className = 'fas fa-microphone';
        
        this.addSystemMessage("Обработка аудио...");
    }
    
    // Process recorded audio data
    async processAudioData() {
        if (!this.state.audioChunks.length) return;
        
        try {
            // Create audio blob
            const audioBlob = new Blob(this.state.audioChunks, { type: 'audio/wav' });
            
            // Create form data for API
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');
            
            // Send to ASR service
            const response = await fetch(`${this.ASR_SERVICE_URL}/transcribe`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.transcript) {
                    // Show transcript in input field
                    this.elements.userInput.value = data.transcript;
                    
                    // Now process the audio for emotion analysis if emotional mode is active
                    if (this.state.emotionalModeActive) {
                        await this.analyzeAudioEmotion(audioBlob);
                    }
                    
                    // Auto-send the message
                    this.sendMessage();
                } else {
                    this.addSystemMessage("Речь не распознана. Пожалуйста, попробуйте снова.", 'error');
                }
            } else {
                this.addSystemMessage("Ошибка при обработке аудио.", 'error');
            }
        } catch (error) {
            console.error("Error processing audio:", error);
            this.addSystemMessage("Не удалось обработать аудио.", 'error');
        }
    }
    
    // Analyze audio for emotional content
    async analyzeAudioEmotion(audioBlob) {
        try {
            // Create form data for API
            const formData = new FormData();
            formData.append('audio', audioBlob, 'emotion.wav');
            
            // Send to emotion analysis endpoint
            const response = await fetch(`${this.API_BASE_URL}/api/analyze_audio_emotion`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.emotion) {
                    // Update emotion state
                    this.state.currentEmotion = data.emotion.dominant;
                    this.updateEmotionValues(data.emotion.scores);
                    this.updateCurrentState();
                    
                    // Update economy metrics based on detected emotion
                    this.updateEconomyBasedOnEmotion(data.emotion.dominant, data.emotion.scores);
                }
            }
        } catch (error) {
            console.error("Error analyzing audio emotion:", error);
            // Don't show error to user as this is a background process
        }
    }
    
    // Update economy based on detected emotion
    updateEconomyBasedOnEmotion(dominantEmotion, scores) {
        // Complex economy rules would apply here
        // For demo purposes, simplified:
        
        let economyUpdates = {
            collectiveParanoia: this.state.economyMetrics.collectiveParanoia,
            emotionalInflation: this.state.economyMetrics.emotionalInflation,
            systemStability: this.state.economyMetrics.systemStability,
            painCapital: this.state.economyMetrics.painCapital,
            painTrend: this.state.economyMetrics.painTrend
        };
        
        // Adjust paranoia based on fear
        economyUpdates.collectiveParanoia = Math.min(100, 
            economyUpdates.collectiveParanoia + (scores.fear * 10) - (scores.happiness * 5));
        
        // Inflation increases with happiness
        economyUpdates.emotionalInflation = Math.min(100, 
            economyUpdates.emotionalInflation + (scores.happiness * 5));
        
        // Stability affected by all emotions but mostly neutral
        economyUpdates.systemStability = Math.min(100, Math.max(0,
            economyUpdates.systemStability + (scores.neutral * 10) - (scores.anger * 5) - (scores.fear * 5)));
        
        // Pain capital increases with negative emotions
        const painGain = (scores.anger * 15) + (scores.sadness * 10) + (scores.fear * 12);
        economyUpdates.painCapital += painGain;
        
        // Pain trend
        economyUpdates.painTrend = painGain > 0 ? painGain : economyUpdates.painTrend;
        
        // Update metrics
        this.updateEconomyMetrics(economyUpdates);
    }
}
