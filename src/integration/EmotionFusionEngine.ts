import { FacialEmotionVector } from '../analyzers/EmotionAnalyzer';
import { VoiceEmotionVector } from '../analyzers/VoiceEmotionAnalyzer';
import { TouchEmotionVector } from '../analyzers/TouchBehaviorAnalyzer';

export interface FusedEmotionVector {
  valence: number;
  arousal: number;
  stress: number;
  confidence: number;
  emotionLabel: string;
}

export type FusionCallback = (data: FusedEmotionVector) => void;

export class EmotionFusionEngine {
  private buffer: Record<'facial' | 'voice' | 'touch', any[]> = {
    facial: [],
    voice: [],
    touch: [],
  };

  constructor(private callback: FusionCallback, private windowMs = 5000) {}

  addFacial(data: FacialEmotionVector) {
    this.buffer.facial.push({ ...data, timestamp: Date.now() } as any);
    this.cleanup();
  }
  addVoice(data: VoiceEmotionVector) {
    this.buffer.voice.push({ ...data, timestamp: Date.now() } as any);
    this.cleanup();
  }
  addTouch(data: TouchEmotionVector) {
    this.buffer.touch.push({ ...data, timestamp: Date.now() } as any);
    this.cleanup();
  }

  private cleanup() {
    const cutoff = Date.now() - this.windowMs;
    (['facial', 'voice', 'touch'] as const).forEach((key) => {
      this.buffer[key] = this.buffer[key].filter((d: any) => d.timestamp >= cutoff);
    });
    this.fuse();
  }

  private fuse() {
    const { facial, voice, touch } = this.buffer;
    if (facial.length === 0 && voice.length === 0 && touch.length === 0) return;

    const weights = { facial: 0.5, voice: 0.3, touch: 0.2 };
    let valence = 0,
      arousal = 0,
      stress = 0,
      confidence = 0,
      totalWeight = 0;

    if (facial.length) {
      const avg = this.avgFacial(facial);
      valence += avg.valence * weights.facial;
      arousal += avg.arousal * weights.facial;
      confidence += avg.confidence * weights.facial;
      totalWeight += weights.facial;
    }
    if (voice.length) {
      const avg = this.avgVoice(voice);
      valence += avg.valence * weights.voice;
      arousal += avg.arousal * weights.voice;
      stress += avg.stress * weights.voice;
      totalWeight += weights.voice;
    }
    if (touch.length) {
      const avg = this.avgTouch(touch);
      stress += avg.stress * weights.touch;
      arousal += (avg.impatience + avg.velocity) * weights.touch;
      totalWeight += weights.touch;
    }

    if (totalWeight > 0) {
      valence /= totalWeight;
      arousal /= totalWeight;
      stress /= totalWeight;
      confidence /= totalWeight;
    }

    const fused: FusedEmotionVector = {
      valence: Math.max(-1, Math.min(1, valence)),
      arousal: Math.max(-1, Math.min(1, arousal)),
      stress: Math.max(0, Math.min(1, stress)),
      confidence: Math.max(0, Math.min(1, confidence)),
      emotionLabel: this.getEmotionLabel(valence, arousal),
    };

    this.callback(fused);
  }

  private avgFacial(arr: FacialEmotionVector[]) {
    const sum = arr.reduce(
      (acc, cur) => {
        acc.valence += cur.valence;
        acc.arousal += cur.arousal;
        acc.confidence += cur.confidence;
        return acc;
      },
      { valence: 0, arousal: 0, confidence: 0 }
    );
    const len = arr.length;
    return { valence: sum.valence / len, arousal: sum.arousal / len, confidence: sum.confidence / len };
  }
  private avgVoice(arr: VoiceEmotionVector[]) {
    const sum = arr.reduce(
      (acc, cur) => {
        acc.valence += cur.valence;
        acc.arousal += cur.arousal;
        acc.stress += cur.stress;
        return acc;
      },
      { valence: 0, arousal: 0, stress: 0 }
    );
    const len = arr.length;
    return { valence: sum.valence / len, arousal: sum.arousal / len, stress: sum.stress / len };
  }
  private avgTouch(arr: TouchEmotionVector[]) {
    const sum = arr.reduce(
      (acc, cur) => {
        acc.stress += cur.stress;
        acc.impatience += cur.impatience;
        acc.velocity += cur.velocity;
        return acc;
      },
      { stress: 0, impatience: 0, velocity: 0 }
    );
    const len = arr.length;
    return { stress: sum.stress / len, impatience: sum.impatience / len, velocity: sum.velocity / len };
  }

  private getEmotionLabel(valence: number, arousal: number) {
    if (valence > 0.3 && arousal > 0.3) return 'excited';
    if (valence > 0.3 && arousal < -0.3) return 'content';
    if (valence < -0.3 && arousal > 0.3) return 'stressed';
    if (valence < -0.3 && arousal < -0.3) return 'depressed';
    if (arousal > 0.3) return 'aroused';
    if (arousal < -0.3) return 'calm';
    if (valence > 0.3) return 'pleasant';
    if (valence < -0.3) return 'unpleasant';
    return 'neutral';
  }
}