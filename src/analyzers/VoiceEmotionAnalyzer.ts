import Meyda, { MeydaFeaturesObject } from 'meyda';

export interface VoiceEmotionVector {
  valence: number; // -1..1
  arousal: number; // -1..1
  stress: number; // 0..1
  confidence: number; // 0..1
  volume: number;
  pitch: number;
  clarity: number;
  energy: number;
  mfcc: number[];
}

export type VoiceCallback = (data: VoiceEmotionVector) => void;

export class VoiceEmotionAnalyzer {
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private microphone: MediaStreamAudioSourceNode | null = null;
  private scriptNode: ScriptProcessorNode | null = null;
  private isRunning = false;

  constructor(private callback: VoiceCallback) {}

  async initialize(sampleRate = 44100) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate,
      },
    });

    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    this.analyser = this.audioContext.createAnalyser();
    this.microphone = this.audioContext.createMediaStreamSource(stream);
    this.scriptNode = this.audioContext.createScriptProcessor(4096, 1, 1);

    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0.3;

    this.microphone.connect(this.analyser);
    this.analyser.connect(this.scriptNode);
    this.scriptNode.connect(this.audioContext.destination);

    this.scriptNode.onaudioprocess = (ev) => this.process(ev);
  }

  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    if (this.audioContext?.state === 'suspended') this.audioContext.resume();
  }

  stop() {
    this.isRunning = false;
    if (this.audioContext?.state === 'running') this.audioContext.suspend();
  }

  private process(ev: AudioProcessingEvent) {
    if (!this.isRunning) return;

    const input = ev.inputBuffer.getChannelData(0);

    // Extract features
    const features: MeydaFeaturesObject = Meyda.extract(
      [
        'rms',
        'mfcc',
        'spectralCentroid',
        'spectralRolloff',
        'zcr',
        'chroma',
        'energy',
      ],
      input
    ) as MeydaFeaturesObject;

    const rms = features.rms ?? 0;
    const spectralCentroid = features.spectralCentroid ?? 0;
    const zcr = features.zcr ?? 0;
    const energy = features.energy ?? 0;

    const arousal = Math.min(1, (rms * 10 + energy * 5) / 2);
    const valence = Math.max(-1, Math.min(1, (spectralCentroid - 1000) / 1000));
    const stress = Math.min(1, (zcr * 100 + (1 - energy)) / 2);
    const confidence = Math.max(0, 1 - Math.abs(rms - 0.3) * 2);

    const vector: VoiceEmotionVector = {
      arousal,
      valence,
      stress,
      confidence,
      volume: rms,
      pitch: spectralCentroid,
      clarity: 1 - zcr,
      energy,
      mfcc: features.mfcc ?? [],
    };

    this.callback(vector);
  }
}