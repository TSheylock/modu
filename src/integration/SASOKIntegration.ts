import { EmotionAnalyzer } from '../analyzers/EmotionAnalyzer';
import { VoiceEmotionAnalyzer } from '../analyzers/VoiceEmotionAnalyzer';
import { TouchBehaviorAnalyzer } from '../analyzers/TouchBehaviorAnalyzer';
import { EmotionFusionEngine, FusedEmotionVector } from './EmotionFusionEngine';

export interface SASOKEvents {
  onEmotion?: (data: FusedEmotionVector) => void;
}

export class SASOKIntegration {
  private emotionAnalyzer: EmotionAnalyzer | null = null;
  private voiceAnalyzer: VoiceEmotionAnalyzer | null = null;
  private touchAnalyzer: TouchBehaviorAnalyzer | null = null;
  private fusion: EmotionFusionEngine;
  private ws: WebSocket | null = null;
  private buffer: any[] = [];
  private sessionId = this.generateId();

  constructor(private events: SASOKEvents = {}) {
    this.fusion = new EmotionFusionEngine((data) => {
      this.send({ type: 'fused_emotion', data });
      this.events.onEmotion?.(data);
    });
  }

  async initialize(video: HTMLVideoElement) {
    // Facial analyzer
    this.emotionAnalyzer = new EmotionAnalyzer(video, '/models', (data) => {
      this.fusion.addFacial(data);
      this.send({ type: 'facial_emotion', data });
    });
    await this.emotionAnalyzer.initialize();

    // Voice analyzer
    this.voiceAnalyzer = new VoiceEmotionAnalyzer((data) => {
      this.fusion.addVoice(data);
      this.send({ type: 'voice_emotion', data });
    });
    await this.voiceAnalyzer.initialize();

    // Touch analyzer
    this.touchAnalyzer = new TouchBehaviorAnalyzer((data) => {
      this.fusion.addTouch(data);
      this.send({ type: 'touch_behavior', data });
    });
    this.touchAnalyzer.initialize();
  }

  start() {
    this.emotionAnalyzer?.start();
    this.voiceAnalyzer?.start();
    this.touchAnalyzer?.startTracking();
    this.connect();
  }
  stop() {
    this.emotionAnalyzer?.stop();
    this.voiceAnalyzer?.stop();
    this.touchAnalyzer?.stopTracking();
    if (this.ws) this.ws.close();
  }

  private connect() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;
    this.ws = new WebSocket('wss://sasok.example.com/realtime');
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.flush();
      this.send({ type: 'handshake', sessionId: this.sessionId });
    };
    this.ws.onclose = () => {
      console.warn('WebSocket closed, retrying in 5s');
      setTimeout(() => this.connect(), 5000);
    };
    this.ws.onerror = (e) => console.error('WebSocket error', e);
    this.ws.onmessage = (e) => this.handleMessage(e.data);
  }

  private send(payload: any) {
    const msg = { sessionId: this.sessionId, timestamp: Date.now(), ...payload };
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    } else {
      this.buffer.push(msg);
    }
  }

  private flush() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.buffer.forEach((m) => this.ws!.send(JSON.stringify(m)));
      this.buffer = [];
    }
  }

  private handleMessage(data: string) {
    try {
      const msg = JSON.parse(data);
      console.log('Message from SASOK', msg);
      // handle specific message types if needed
    } catch (err) {
      console.error('Invalid JSON from SASOK', err);
    }
  }

  private generateId() {
    return 'xxxxxxxx'.replace(/[x]/g, () => ((Math.random() * 16) | 0).toString(16));
  }
}