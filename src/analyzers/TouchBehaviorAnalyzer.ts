export interface TouchEmotionVector {
  stress: number;
  calm: number;
  focus: number;
  impatience: number;
  velocity: number;
  pressure: number;
  frequency: number;
}

export type TouchCallback = (data: TouchEmotionVector) => void;

export class TouchBehaviorAnalyzer {
  private touchHistory: { x: number; y: number; timestamp: number; pressure: number }[] = [];
  private clickHistory: { timestamp: number }[] = [];
  private isTracking = false;

  constructor(private callback: TouchCallback) {}

  initialize() {
    document.addEventListener('touchstart', (e) => this.handleTouch(e));
    document.addEventListener('click', () => this.handleClick());
  }

  startTracking() {
    this.isTracking = true;
  }

  stopTracking() {
    this.isTracking = false;
  }

  private handleTouch(event: TouchEvent) {
    if (!this.isTracking) return;
    const t = event.touches[0];
    const point = {
      x: t.clientX,
      y: t.clientY,
      timestamp: Date.now(),
      pressure: (t as any).force ?? 1,
    };
    this.touchHistory.push(point);
    if (this.touchHistory.length > 100) this.touchHistory.shift();
    this.analyze();
  }

  private handleClick() {
    if (!this.isTracking) return;
    this.clickHistory.push({ timestamp: Date.now() });
    if (this.clickHistory.length > 100) this.clickHistory.shift();
    this.analyze();
  }

  private analyze() {
    const touches = this.touchHistory;
    if (touches.length < 2) return;

    const recent = touches[touches.length - 1];
    const prev = touches[touches.length - 2];
    const distance = Math.hypot(recent.x - prev.x, recent.y - prev.y);
    const velocity = distance / (recent.timestamp - prev.timestamp + 1);
    const pressure = recent.pressure;
    const frequency = this.touchFrequency();

    const stress = Math.min(1, velocity * 0.1 + pressure * 0.5 + frequency * 0.3);
    const calm = Math.max(0, 1 - stress);
    const focus = Math.max(0, 1 - Math.abs(velocity - 0.5) - Math.abs(pressure - 0.5));
    const impatience = Math.min(1, frequency * 2);

    const vector: TouchEmotionVector = {
      stress,
      calm,
      focus,
      impatience,
      velocity,
      pressure,
      frequency,
    };
    this.callback(vector);
  }

  private touchFrequency() {
    if (this.touchHistory.length < 2) return 0;
    const timeSpan = this.touchHistory[this.touchHistory.length - 1].timestamp - this.touchHistory[0].timestamp;
    return timeSpan > 0 ? (this.touchHistory.length / timeSpan) * 1000 : 0;
  }
}