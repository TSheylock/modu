import * as faceapi from 'face-api.js';

export interface BaseEmotions {
  angry: number;
  disgusted: number;
  fearful: number;
  happy: number;
  neutral: number;
  sad: number;
  surprised: number;
}

export interface FacialEmotionVector extends BaseEmotions {
  valence: number; // -1..1
  arousal: number; // -1..1
  eyeAspectRatio?: number;
  mouthAspectRatio?: number;
  browHeight?: number;
  confidence: number; // 0..1
}

export type EmotionCallback = (data: FacialEmotionVector) => void;

export class EmotionAnalyzer {
  private isRunning = false;
  private intervalId: number | null = null;

  constructor(
    private video: HTMLVideoElement,
    private modelPath = '/models',
    private callback: EmotionCallback
  ) {}

  /** Load face-api models */
  async initialize() {
    await faceapi.nets.tinyFaceDetector.loadFromUri(this.modelPath);
    await faceapi.nets.faceExpressionNet.loadFromUri(this.modelPath);
    await faceapi.nets.faceLandmark68Net.loadFromUri(this.modelPath);
  }

  start(fps = 15) {
    if (this.isRunning) return;
    this.isRunning = true;
    const interval = 1000 / fps;
    this.intervalId = window.setInterval(() => this.detect(), interval);
  }

  stop() {
    if (this.intervalId) window.clearInterval(this.intervalId);
    this.isRunning = false;
  }

  private async detect() {
    if (!this.video || this.video.readyState < 2) return;

    const detections = await faceapi
      .detectAllFaces(this.video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions();

    if (!detections || detections.length === 0) return;

    const det = detections[0];
    const expr = det.expressions as BaseEmotions;

    const valence = (expr.happy + expr.surprised) - (expr.sad + expr.angry + expr.fearful);
    const arousal = (expr.angry + expr.fearful + expr.surprised) - (expr.sad + expr.neutral);

    const vector: FacialEmotionVector = {
      ...expr,
      valence: Math.max(-1, Math.min(1, valence)),
      arousal: Math.max(-1, Math.min(1, arousal)),
      confidence: Math.max(...Object.values(expr)),
    };

    this.callback(vector);
  }
}