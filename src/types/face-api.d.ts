declare module 'face-api.js' {
  // We only need minimal typings for compilation purpose.
  // Consumers can import * as faceapi from 'face-api.js';
  export const nets: any;
  export class TinyFaceDetectorOptions {
    constructor(inputSize?: number, scoreThreshold?: number);
  }
  export interface FaceExpressions {
    angry: number;
    disgusted: number;
    fearful: number;
    happy: number;
    neutral: number;
    sad: number;
    surprised: number;
  }
  export interface FaceLandmarks68 {}
  export interface FullFaceDescription {
    expressions: FaceExpressions;
    landmarks: FaceLandmarks68;
  }
  export function detectAllFaces(
    input: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    options?: TinyFaceDetectorOptions
  ): any;
  export function withFaceLandmarks(): any;
  export function withFaceExpressions(): any;
  export function loadFromUri(uri: string): Promise<void>;
}

declare module 'meyda' {
  export interface MeydaFeaturesObject {
    rms?: number;
    mfcc?: number[];
    spectralCentroid?: number;
    spectralRolloff?: number;
    zcr?: number;
    chroma?: number[];
    energy?: number;
  }
  export function extract(
    featureNames: string[],
    signal: Float32Array,
    options?: any
  ): MeydaFeaturesObject;
  export const MeydaAnalyzer: any;
  const Meyda: any;
  export default Meyda;
}