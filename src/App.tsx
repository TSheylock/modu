import React, { useEffect, useRef, useState } from 'react';
import './App.css';

// Dynamic imports for browser-only libraries
const loadFaceApi = () => import('face-api.js');
const loadMeyda = () => import('meyda');

interface EmotionData {
  valence: number;
  arousal: number;
  stress: number;
  confidence: number;
  emotionLabel: string;
}

// -----------------------------------------------------------------------------
// Helper: map value (-1..1) to percentage (0..100)
// -----------------------------------------------------------------------------
const toPercent = (v: number) => `${((v + 1) * 50).toFixed(0)}%`;

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [emotion, setEmotion] = useState<EmotionData | null>(null);

  // Initialize camera stream
  useEffect(() => {
    const initCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error('Error accessing camera', err);
      }
    };
    initCamera();
  }, []);

  // Dummy timer to update emotion state every second (placeholder)
  useEffect(() => {
    const id = setInterval(() => {
      // random values for demo
      setEmotion({
        valence: Math.random() * 2 - 1,
        arousal: Math.random() * 2 - 1,
        stress: Math.random(),
        confidence: Math.random(),
        emotionLabel: 'neutral',
      });
    }, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-4">
      <h1 className="text-2xl font-bold mb-4">SASOK Real-time Emotion Capture (Demo)</h1>
      <div className="relative border-2 border-gray-800 rounded-lg overflow-hidden w-[640px] h-[480px]">
        <video ref={videoRef} autoPlay muted className="w-full h-full object-cover" />
      </div>

      {emotion && (
        <div className="bg-white rounded-lg p-6 shadow-md mt-6 w-full max-w-lg">
          <h2 className="font-semibold text-lg mb-3">Текущие эмоции</h2>
          <Metric label="Валентность" value={emotion.valence} color="bg-green-500" />
          <Metric label="Возбуждение" value={emotion.arousal} color="bg-orange-500" />
          <Metric label="Стресс" value={emotion.stress * 2 - 1} color="bg-red-500" />
          <Metric label="Уверенность" value={emotion.confidence * 2 - 1} color="bg-blue-500" />
          <div className="mt-3 font-bold">Эмоция: {emotion.emotionLabel}</div>
        </div>
      )}
    </div>
  );
};

interface MetricProps {
  label: string;
  value: number; // -1..1
  color: string;
}

const Metric: React.FC<MetricProps> = ({ label, value, color }) => {
  return (
    <div className="my-2">
      <div className="text-sm">{label}</div>
      <div className="w-full h-4 bg-gray-200 rounded overflow-hidden">
        <div
          className={`${color} h-full transition-all duration-300`}
          style={{ width: toPercent(value) }}
        />
      </div>
    </div>
  );
};

export default App;