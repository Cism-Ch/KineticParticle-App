export enum ParticleMode {
  FLOAT = 'float',
  GATHER = 'gather',
  SWIRL = 'swirl',
  EXPLODE = 'explode',
  RAIN = 'rain'
}

export interface ParticleConfig {
  mode: ParticleMode;
  color: string;
  speed: number;
  count: number;
  size: number;
}

export interface LogEntry {
  timestamp: number;
  message: string;
  source: 'system' | 'gemini' | 'user';
}
