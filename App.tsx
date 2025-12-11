import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import { GoogleGenAI, LiveServerMessage, Modality, FunctionDeclaration, Type } from '@google/genai';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { Activity, Mic, MicOff, Video, VideoOff, Zap, Hand, Scaling, CloudRain, Cpu } from 'lucide-react';

import { ParticleScene } from './components/ParticleScene';
import { ParticleConfig, ParticleMode, LogEntry } from './types';
import { createPcmBlob, decodeAudioData, base64ToUint8Array } from './utils/audioUtils';

// --- Constants ---
const GESTURE_CONFIDENCE_THRESHOLD = 0.6;

// --- Gemini Tool Definitions (Voice Only Now) ---
const setParticleColorDecl: FunctionDeclaration = {
  name: 'setParticleColor',
  parameters: {
    type: Type.OBJECT,
    description: 'Trigger this when the user asks to change the color.',
    properties: {
      color: {
        type: Type.STRING,
        description: 'Hex color code or standard color name.',
      },
    },
    required: ['color'],
  },
};

const setParticleSpeedDecl: FunctionDeclaration = {
  name: 'setParticleSpeed',
  parameters: {
    type: Type.OBJECT,
    description: 'Updates the animation speed based on voice command.',
    properties: {
      speed: {
        type: Type.NUMBER,
        description: 'Speed multiplier, typically between 0.1 and 5.0.',
      },
    },
    required: ['speed'],
  },
};

const SYSTEM_INSTRUCTION = `
You are a Voice Command Interface for a Kinetic Particle System.
**CRITICAL: DO NOT ATTEMPT TO RECOGNIZE GESTURES.** 
Gestures are handled by a local dedicated vision model.
Your role is purely to listen to the user's voice and:
1. Change colors when asked (e.g. "Make it purple", "Red").
2. Change speed when asked (e.g. "Faster", "Slow down").
3. Answer questions about the system briefly.

Keep responses extremely short and robotic.
`;

export default function App() {
  // --- State ---
  const [config, setConfig] = useState<ParticleConfig>({
    mode: ParticleMode.FLOAT,
    color: '#00ffff',
    speed: 1,
    count: 2000,
    size: 1,
  });
  const [active, setActive] = useState(false);
  const [visionReady, setVisionReady] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isMicOn, setIsMicOn] = useState(true);
  const [currentGesture, setCurrentGesture] = useState<string>('None');

  // --- Refs ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<any>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const requestRef = useRef<number>();
  const isMicOnRef = useRef(isMicOn);
  
  // Audio Refs
  const inputAudioCtxRef = useRef<AudioContext | null>(null);
  const outputAudioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  // --- Effects ---
  useEffect(() => { isMicOnRef.current = isMicOn; }, [isMicOn]);

  useEffect(() => {
    // Initialize MediaPipe Vision
    const initVision = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm"
        );
        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1
        });
        handLandmarkerRef.current = handLandmarker;
        setVisionReady(true);
        addLog("Local Vision Model Loaded", "system");
      } catch (e) {
        console.error("Vision init failed", e);
        addLog("Vision Model Failed to Load", "system");
      }
    };
    initVision();

    return () => {
       stopSession();
       if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, []);

  const addLog = useCallback((message: string, source: LogEntry['source']) => {
    setLogs(prev => [...prev.slice(-4), { timestamp: Date.now(), message, source }]);
  }, []);

  // --- MediaPipe Gesture Logic ---
  const predictWebcam = () => {
     // Ensure video is ready before predicting
     if (!handLandmarkerRef.current || !videoRef.current || videoRef.current.readyState < 2) {
         requestRef.current = requestAnimationFrame(predictWebcam);
         return;
     }

     const nowInMs = Date.now();
     const results = handLandmarkerRef.current.detectForVideo(videoRef.current, nowInMs);

     if (results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];
        analyzeGesture(landmarks);
     } else {
        // No hand detected
     }
     
     requestRef.current = requestAnimationFrame(predictWebcam);
  };

  const analyzeGesture = (landmarks: any[]) => {
      // Finger Indices
      // Thumb: 4, Index: 8, Middle: 12, Ring: 16, Pinky: 20
      // PIP Joints (Knuckles approx): 2, 6, 10, 14, 18
      
      const thumbTip = landmarks[4];
      const indexTip = landmarks[8];
      const middleTip = landmarks[12];
      const ringTip = landmarks[16];
      const pinkyTip = landmarks[20];
      
      const indexBase = landmarks[5];
      const middleBase = landmarks[9];
      const ringBase = landmarks[13];
      const pinkyBase = landmarks[17];

      // Helper: Is finger extended? (Tip y < Base y) - Note: Y is normalized 0-1, 0 is top.
      const isIndexUp = indexTip.y < indexBase.y;
      const isMiddleUp = middleTip.y < middleBase.y;
      const isRingUp = ringTip.y < ringBase.y;
      const isPinkyUp = pinkyTip.y < pinkyBase.y;

      // Pinch Detection (Euclidean distance between thumb and index)
      const pinchDist = Math.sqrt(
          Math.pow(thumbTip.x - indexTip.x, 2) + Math.pow(thumbTip.y - indexTip.y, 2)
      );

      let detectedMode = config.mode;
      let detectedName = "";

      if (pinchDist < 0.05) {
          // PINCH
          detectedName = "PINCH (Resize)";
          setConfig(prev => ({ ...prev, size: 0.5 }));
      } else if (!isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp) {
          // FIST
          detectedName = "FIST (Gather)";
          detectedMode = ParticleMode.GATHER;
      } else if (isIndexUp && isMiddleUp && !isRingUp && !isPinkyUp) {
          // PEACE
          detectedName = "PEACE (Rain)";
          detectedMode = ParticleMode.RAIN;
      } else if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp) {
          // POINT
          detectedName = "POINT (Swirl)";
          detectedMode = ParticleMode.SWIRL;
      } else if (isIndexUp && isMiddleUp && isRingUp && isPinkyUp) {
          // OPEN PALM
          detectedName = "OPEN PALM (Explode)";
          detectedMode = ParticleMode.EXPLODE;
          setConfig(prev => ({ ...prev, size: 1.0 })); // Reset size on open
      } else {
          detectedName = "NEUTRAL (Float)";
          detectedMode = ParticleMode.FLOAT;
      }

      if (detectedMode !== config.mode) {
          setConfig(prev => ({ ...prev, mode: detectedMode }));
          setCurrentGesture(detectedName);
      }
      
      // Update HUD if name changes
      if (currentGesture !== detectedName && detectedName !== "") {
          setCurrentGesture(detectedName);
      }
  };

  // --- Gemini Connection ---
  const connectToGemini = async () => {
    if (active) return;
    if (!visionReady) {
        addLog("Wait for Vision Model...", "system");
        return;
    }
    
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      // Initialize Audio Contexts
      inputAudioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      outputAudioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      
      // CRITICAL: Resume audio contexts to prevent "Network error" / stalled state
      await inputAudioCtxRef.current.resume();
      await outputAudioCtxRef.current.resume();

      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: { channelCount: 1, sampleRate: 16000 }, 
        video: { width: 640, height: 480, frameRate: 30 } 
      });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().then(() => {
             // Start Vision Loop
             requestRef.current = requestAnimationFrame(predictWebcam);
        });
      }

      const config = {
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        config: {
          systemInstruction: SYSTEM_INSTRUCTION,
          tools: [{ functionDeclarations: [setParticleColorDecl, setParticleSpeedDecl] }],
          responseModalities: [Modality.AUDIO], 
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
          },
        },
        callbacks: {
          onopen: () => {
            addLog('Gemini Voice Connected', 'system');
            setActive(true);
            // Delay slightly to ensure socket is ready
            setTimeout(() => startAudioProcessing(stream), 100);
          },
          onmessage: async (message: LiveServerMessage) => {
             if (message.toolCall) {
               addLog(`Voice Cmd: ${message.toolCall.functionCalls.map(f => f.name).join(', ')}`, 'gemini');
               handleToolCalls(message.toolCall);
             }
             const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
             if (base64Audio && outputAudioCtxRef.current) {
                playAudioChunk(base64Audio);
             }
          },
          onclose: () => {
            addLog('Connection closed', 'system');
            stopSession();
          },
          onerror: (err: any) => {
            console.error(err);
            if (active) {
                addLog(`Error: ${err.message || 'Network error'}`, 'system');
                stopSession();
            }
          }
        }
      };

      const sessionPromise = ai.live.connect(config);
      sessionRef.current = sessionPromise;

    } catch (e: any) {
      addLog(`Failed to connect: ${e.message}`, 'system');
      stopSession();
    }
  };

  const handleToolCalls = (toolCall: any) => {
      const responses: any[] = [];
      for (const call of toolCall.functionCalls) {
          const { name, args, id } = call;
          let result = { status: 'ok' };
          
          if (name === 'setParticleColor') {
              setConfig(prev => ({ ...prev, color: args.color }));
              addLog(`Color: ${args.color}`, 'system');
          } else if (name === 'setParticleSpeed') {
              setConfig(prev => ({ ...prev, speed: args.speed }));
              addLog(`Speed: ${args.speed}`, 'system');
          } 
          responses.push({ id, name, response: { result } });
      }

      if (sessionRef.current) {
          sessionRef.current.then((session: any) => {
              try { session.sendToolResponse({ functionResponses: responses }); } catch (e) {}
          });
      }
  };

  const startAudioProcessing = (stream: MediaStream) => {
      if (inputAudioCtxRef.current && inputAudioCtxRef.current.state !== 'closed') {
          const ctx = inputAudioCtxRef.current;
          const source = ctx.createMediaStreamSource(stream);
          // Use 2048 buffer for lower latency/stability trade-off
          const processor = ctx.createScriptProcessor(2048, 1, 1);
          
          processor.onaudioprocess = (e) => {
              if (!isMicOnRef.current || !sessionRef.current) return;
              
              const inputData = e.inputBuffer.getChannelData(0);
              // Pass context sample rate to ensure mimeType matches data
              const pcmBlob = createPcmBlob(inputData, ctx.sampleRate);
              
              const currentSessionPromise = sessionRef.current;
              
              currentSessionPromise.then((session: any) => {
                  if (sessionRef.current !== currentSessionPromise) return;
                  try { session.sendRealtimeInput({ media: pcmBlob }); } catch (e) {}
              }).catch(() => {});
          };
          
          source.connect(processor);
          processor.connect(ctx.destination);
          processorRef.current = processor;
      }
  };

  const playAudioChunk = async (base64Audio: string) => {
      const ctx = outputAudioCtxRef.current;
      if (!ctx || ctx.state === 'closed') return;
      try {
        nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
        const audioBuffer = await decodeAudioData(base64ToUint8Array(base64Audio), ctx);
        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);
        source.addEventListener('ended', () => sourcesRef.current.delete(source));
        source.start(nextStartTimeRef.current);
        nextStartTimeRef.current += audioBuffer.duration;
        sourcesRef.current.add(source);
      } catch (err) { console.error("Audio decode error", err); }
  };

  const stopSession = () => {
    sessionRef.current = null;
    setActive(false);
    
    // Cleanup script processor
    if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
    }

    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    if (inputAudioCtxRef.current) inputAudioCtxRef.current.close().catch(console.error);
    if (outputAudioCtxRef.current) outputAudioCtxRef.current.close().catch(console.error);
  };

  const toggleMic = () => setIsMicOn(prev => !prev);

  // --- Render ---
  return (
    <div className="relative w-full h-screen bg-black overflow-hidden font-sans">
      
      {/* 3D Scene Background */}
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [0, 0, 40], fov: 60 }}>
          <color attach="background" args={['#050505']} />
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
          <ParticleScene config={config} />
          <OrbitControls makeDefault enableZoom={false} autoRotate={config.mode === ParticleMode.SWIRL} autoRotateSpeed={2} />
        </Canvas>
      </div>

      {/* UI Overlay */}
      <div className="absolute inset-0 z-10 pointer-events-none flex flex-col justify-between p-6">
        
        {/* Header */}
        <div className="flex justify-between items-start pointer-events-auto">
          <div>
            <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-purple-600 tracking-tighter">
              KINETIC CORE
            </h1>
            <p className="text-cyan-500/70 text-sm mt-1">HYBRID: MEDIAPIPE VISION + GEMINI VOICE</p>
          </div>
          
          <div className="flex gap-4 items-center">
             <div className={`px-3 py-1 rounded-full border ${active ? 'border-green-500/50 bg-green-500/10 text-green-400' : 'border-red-500/50 bg-red-500/10 text-red-400'} text-xs font-mono uppercase transition-all duration-300`}>
                 {active ? 'SYSTEM ONLINE' : 'SYSTEM OFFLINE'}
             </div>
          </div>
        </div>

        {/* Center Start Button */}
        {!active && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-auto bg-black/40 backdrop-blur-sm z-50 flex-col gap-4">
                <button 
                    onClick={connectToGemini}
                    disabled={!visionReady}
                    className={`group relative px-8 py-4 bg-transparent border ${visionReady ? 'border-cyan-500/50 text-cyan-400 hover:text-white hover:border-cyan-400 hover:bg-cyan-500/20' : 'border-gray-700 text-gray-600'} transition-all duration-300 rounded-lg overflow-hidden`}
                >
                    <span className="relative flex items-center gap-3 text-lg font-bold tracking-widest">
                        <Zap className="w-5 h-5" /> {visionReady ? "INITIALIZE HYBRID CORE" : "LOADING VISION MODEL..."}
                    </span>
                </button>
                {!visionReady && <span className="text-xs text-white/40 animate-pulse">Downloading MediaPipe WASM...</span>}
            </div>
        )}

        {/* HUD & Controls */}
        <div className="flex justify-between items-end pointer-events-auto">
            
            {/* Status Panel */}
            <div className="w-64 space-y-2">
                <div className="bg-black/50 backdrop-blur-md border border-white/10 p-4 rounded-lg">
                    <h3 className="text-xs text-white/50 uppercase mb-2 border-b border-white/10 pb-1">Telemetry</h3>
                    <div className="space-y-1 font-mono text-xs">
                         <div className="flex justify-between"><span className="text-white/70">VISION</span> <span className="text-green-400">LOCAL (MP)</span></div>
                        <div className="flex justify-between"><span className="text-white/70">MODE</span> <span className="text-cyan-400">{config.mode.toUpperCase()}</span></div>
                        <div className="flex justify-between"><span className="text-white/70">GESTURE</span> <span className="text-yellow-400">{currentGesture}</span></div>
                        <div className="flex justify-between"><span className="text-white/70">SPEED</span> <span className="text-cyan-400">x{config.speed.toFixed(1)}</span></div>
                        <div className="flex justify-between"><span className="text-white/70">SIZE</span> <span className="text-cyan-400">x{config.size.toFixed(1)}</span></div>
                    </div>
                </div>

                {/* Log Feed */}
                <div className="bg-black/50 backdrop-blur-md border border-white/10 p-4 rounded-lg h-32 overflow-hidden flex flex-col justify-end">
                    <div className="space-y-1">
                        {logs.map((log, i) => (
                            <div key={log.timestamp + i} className="text-[10px] font-mono opacity-70 animate-in slide-in-from-left-2 fade-in duration-300">
                                <span className={log.source === 'system' ? 'text-yellow-500' : 'text-purple-400'}>
                                    [{log.source.toUpperCase()}]
                                </span> {log.message}
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Media Controls */}
            <div className="flex flex-col gap-2 items-end">
                {/* Hidden Video for stream capture - Used by MediaPipe */}
                <video ref={videoRef} className="hidden" muted playsInline />

                {/* Video Preview (Mini) with Landmarks Overlay */}
                <div className="w-48 h-36 bg-black/80 border border-white/10 rounded-lg overflow-hidden relative">
                    {/* We just show the raw video for preview, simple ref sync */}
                     <VideoPreview stream={streamRef.current} />
                     
                     <div className="absolute bottom-2 right-2 flex gap-1 z-20">
                        <button onClick={toggleMic} className={`p-2 rounded-full ${isMicOn ? 'bg-white/10 text-white' : 'bg-red-500/20 text-red-500'} hover:bg-white/20 transition-colors`}>
                            {isMicOn ? <Mic size={14} /> : <MicOff size={14} />}
                        </button>
                     </div>
                     <div className="absolute top-2 left-2 flex gap-1 z-20">
                        <span className="text-[10px] bg-green-900/50 text-green-400 px-2 py-0.5 rounded border border-green-500/30 flex items-center gap-1">
                            <Cpu size={10} /> VISION ACTIVE
                        </span>
                     </div>
                </div>

                <div className="bg-black/50 backdrop-blur-md border border-white/10 p-4 rounded-lg text-xs text-white/60 w-52">
                    <p className="mb-2 uppercase tracking-wider text-xs border-b border-white/10 pb-1"><strong className="text-white">Gesture Protocol</strong></p>
                    <ul className="space-y-1.5">
                        <li className="flex items-center gap-2"><div className="w-4 flex justify-center"><Hand size={12}/></div> Open Palm (Explode)</li>
                        <li className="flex items-center gap-2"><div className="w-4 flex justify-center"><div className="w-2 h-2 bg-white rounded-full"></div></div> Fist (Gather)</li>
                        <li className="flex items-center gap-2"><div className="w-4 flex justify-center"><Zap size={12}/></div> Point (Swirl)</li>
                         <li className="flex items-center gap-2"><div className="w-4 flex justify-center"><CloudRain size={12}/></div> Peace Sign (Rain)</li>
                        <li className="flex items-center gap-2"><div className="w-4 flex justify-center"><Scaling size={12}/></div> Pinch (Shrink)</li>
                    </ul>
                </div>
            </div>
        </div>
      </div>
    </div>
  );
}

// Simple helper to render the video stream in the preview box
const VideoPreview = ({ stream }: { stream: MediaStream | null }) => {
    const ref = useRef<HTMLVideoElement>(null);
    useEffect(() => {
        if (ref.current && stream) {
            ref.current.srcObject = stream;
        }
    }, [stream]);
    return <video ref={ref} autoPlay muted className="w-full h-full object-cover opacity-80 transform scale-x-[-1]" />;
};