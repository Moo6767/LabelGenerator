import { useState, useRef, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, Trash2, Scissors, Check, Video, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, X, SkipBack, SkipForward } from "lucide-react";
import { toast } from "sonner";

interface ClipMarker {
  id: string;
  startTime: number;
  endTime: number;
  label?: string;
}

interface ManualClipData {
  name: string;
  frames: { file: File; url: string; frameNumber: number }[];
  sourceVideo: string;
}

interface ManualClipEditorProps {
  framesPerClip: number;
  frameInterval: number;
  confidenceThreshold: number;
  setConfidenceThreshold: (v: number) => void;
  boundingBoxPadding: number;
  setBoundingBoxPadding: (v: number) => void;
  filterMotionBlur: boolean;
  setFilterMotionBlur: (v: boolean) => void;
  blurThreshold: number;
  setBlurThreshold: (v: number) => void;
  enhanceVideo: boolean;
  setEnhanceVideo: (v: boolean) => void;
  enhanceScale: number;
  setEnhanceScale: (v: number) => void;
  onClipsCreated: (clips: ManualClipData[]) => void;
}

export const ManualClipEditor = ({ 
  framesPerClip, 
  frameInterval, 
  confidenceThreshold,
  setConfidenceThreshold,
  boundingBoxPadding,
  setBoundingBoxPadding,
  filterMotionBlur,
  setFilterMotionBlur,
  blurThreshold,
  setBlurThreshold,
  enhanceVideo,
  setEnhanceVideo,
  enhanceScale,
  setEnhanceScale,
  onClipsCreated 
}: ManualClipEditorProps) => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [markers, setMarkers] = useState<ClipMarker[]>([]);
  const [isExtracting, setIsExtracting] = useState(false);
  const [extractionProgress, setExtractionProgress] = useState(0);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [timelineOffset, setTimelineOffset] = useState(0);
  
  // Pending marker start time for two-click marker creation
  const [pendingMarkerStart, setPendingMarkerStart] = useState<number | null>(null);
  
  // Playback speed state
  const [playbackSpeed, setPlaybackSpeed] = useState<1 | 2>(1);
  
  // Smooth frame stepping state
  const [isKeyHeld, setIsKeyHeld] = useState(false);
  const keyHeldDirection = useRef<1 | -1 | null>(null);
  const frameRequestRef = useRef<number | null>(null);
  const lastFrameTime = useRef<number>(0);
  const stepAccumulator = useRef<number>(0);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);

  // Handle video upload
  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Cleanup old URL
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
      const url = URL.createObjectURL(file);
      setVideoFile(file);
      setVideoUrl(url);
      setMarkers([]);
      setCurrentTime(0);
      setZoomLevel(1);
      setTimelineOffset(0);
      toast.success(`Video geladen: ${file.name}`);
    }
    // Reset input
    if (e.target) e.target.value = "";
  };

  // Video loaded metadata
  const handleVideoLoaded = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  // Video time update
  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  // Play/Pause toggle
  const togglePlayPause = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  // Toggle playback speed between 1x and 2x
  const togglePlaybackSpeed = () => {
    const newSpeed = playbackSpeed === 1 ? 2 : 1;
    setPlaybackSpeed(newSpeed);
    if (videoRef.current) {
      videoRef.current.playbackRate = newSpeed;
    }
  };

  // Apply playback speed when video loads or speed changes
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = playbackSpeed;
    }
  }, [playbackSpeed, videoUrl]);

  // Add a marker at current time
  const addMarker = () => {
    if (duration === 0) return;
    
    // Calculate a reasonable clip duration based on settings
    const clipDuration = framesPerClip * frameInterval;
    const startTime = currentTime;
    const endTime = Math.min(currentTime + clipDuration, duration);
    
    const newMarker: ClipMarker = {
      id: Date.now().toString(),
      startTime,
      endTime,
    };
    
    setMarkers(prev => [...prev, newMarker].sort((a, b) => a.startTime - b.startTime));
    toast.success(`Clip-Marker hinzugefügt: ${formatTime(startTime)} - ${formatTime(endTime)}`);
  };

  // Delete a marker
  const deleteMarker = (id: string) => {
    setMarkers(prev => prev.filter(m => m.id !== id));
    toast.success("Marker gelöscht");
  };

  // Update marker times
  const updateMarker = (id: string, field: "startTime" | "endTime", value: number) => {
    setMarkers(prev => prev.map(m => {
      if (m.id === id) {
        const updated = { ...m, [field]: Math.max(0, Math.min(value, duration)) };
        // Ensure start < end
        if (updated.startTime >= updated.endTime) {
          if (field === "startTime") {
            updated.endTime = Math.min(updated.startTime + 1, duration);
          } else {
            updated.startTime = Math.max(updated.endTime - 1, 0);
          }
        }
        return updated;
      }
      return m;
    }));
  };

  // Seek video
  const seekTo = (time: number) => {
    if (videoRef.current) {
      const clampedTime = Math.max(0, Math.min(time, duration));
      videoRef.current.currentTime = clampedTime;
      setCurrentTime(clampedTime);
    }
  };

  // Frame step (assumes ~30fps, so 1 frame ≈ 0.033s)
  const frameStep = 1 / 30; // ~33ms per frame
  
  // Instant single frame step - directly manipulates video without waiting for state
  const stepFrameInstant = useCallback((direction: 1 | -1) => {
    if (!videoRef.current) return;
    // Pause video when stepping
    if (videoRef.current.paused === false) {
      videoRef.current.pause();
      setIsPlaying(false);
    }
    const newTime = Math.max(0, Math.min(videoRef.current.currentTime + (direction * frameStep), duration));
    videoRef.current.currentTime = newTime;
    // Don't call setCurrentTime here - let the timeupdate event handle it for smoother updates
  }, [duration, frameStep]);

  // Continuous frame stepping loop for held keys - runs at 60fps for smooth scrubbing
  const continuousFrameStep = useCallback((timestamp: number) => {
    if (!videoRef.current || keyHeldDirection.current === null) return;
    
    const deltaTime = timestamp - lastFrameTime.current;
    lastFrameTime.current = timestamp;
    
    // Accumulate time for frame steps - faster stepping: 10 frames per second base, accelerates over time
    const framesPerSecond = Math.min(30, 10 + stepAccumulator.current * 2); // Accelerate from 10 to 30 fps
    stepAccumulator.current += deltaTime / 1000;
    
    const timeStep = (deltaTime / 1000) * framesPerSecond * frameStep;
    const newTime = Math.max(0, Math.min(
      videoRef.current.currentTime + (keyHeldDirection.current * timeStep),
      duration
    ));
    
    videoRef.current.currentTime = newTime;
    setCurrentTime(newTime);
    
    // Continue loop
    frameRequestRef.current = requestAnimationFrame(continuousFrameStep);
  }, [duration, frameStep]);

  // Start continuous stepping
  const startContinuousStep = useCallback((direction: 1 | -1) => {
    if (frameRequestRef.current) return; // Already running
    
    if (videoRef.current && !videoRef.current.paused) {
      videoRef.current.pause();
      setIsPlaying(false);
    }
    
    keyHeldDirection.current = direction;
    stepAccumulator.current = 0;
    lastFrameTime.current = performance.now();
    setIsKeyHeld(true);
    
    // Do one immediate step
    stepFrameInstant(direction);
    
    // Start continuous loop after a short delay (for key repeat feel)
    setTimeout(() => {
      if (keyHeldDirection.current === direction) {
        frameRequestRef.current = requestAnimationFrame(continuousFrameStep);
      }
    }, 150);
  }, [stepFrameInstant, continuousFrameStep]);

  // Stop continuous stepping
  const stopContinuousStep = useCallback(() => {
    if (frameRequestRef.current) {
      cancelAnimationFrame(frameRequestRef.current);
      frameRequestRef.current = null;
    }
    keyHeldDirection.current = null;
    stepAccumulator.current = 0;
    setIsKeyHeld(false);
  }, []);

  // Snap time to grid (0.5 second intervals)
  const snapToGrid = (time: number): number => {
    const snapInterval = 0.5; // Snap to every 0.5 seconds
    return Math.round(time / snapInterval) * snapInterval;
  };

  // Format time as MM:SS.ss
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toFixed(1).padStart(4, "0")}`;
  };

  // Format time short (just seconds)
  const formatTimeShort = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    if (mins > 0) {
      return `${mins}:${secs.toString().padStart(2, "0")}`;
    }
    return `${secs}s`;
  };

  // Handle timeline click - TWO CLICK MARKER CREATION with SNAPPING
  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!timelineRef.current || duration === 0) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = x / rect.width;
    
    // Calculate time based on zoom and offset
    const visibleDuration = duration / zoomLevel;
    const rawTime = Math.max(0, Math.min(timelineOffset + ratio * visibleDuration, duration));
    
    // Snap to grid
    const clickedTime = snapToGrid(rawTime);
    
    // If no pending start, set this as start
    if (pendingMarkerStart === null) {
      setPendingMarkerStart(clickedTime);
      seekTo(clickedTime);
      toast.info(`Start gesetzt: ${formatTime(clickedTime)} - Klicke für Ende`);
    } else {
      // Second click - create marker
      const startTime = Math.min(pendingMarkerStart, clickedTime);
      const endTime = Math.max(pendingMarkerStart, clickedTime);
      
      if (endTime - startTime < 0.5) {
        toast.error("Clip muss mindestens 0.5 Sekunden lang sein");
        setPendingMarkerStart(null);
        return;
      }
      
      const newMarker: ClipMarker = {
        id: Date.now().toString(),
        startTime,
        endTime,
      };
      
      setMarkers(prev => [...prev, newMarker].sort((a, b) => a.startTime - b.startTime));
      setPendingMarkerStart(null);
      seekTo(endTime);
      toast.success(`Clip erstellt: ${formatTime(startTime)} - ${formatTime(endTime)}`);
    }
  };
  
  // Cancel pending marker with Escape
  const cancelPendingMarker = () => {
    if (pendingMarkerStart !== null) {
      setPendingMarkerStart(null);
      toast.info("Marker-Erstellung abgebrochen");
    }
  };

  // Calculate visible timeline range
  const getVisibleRange = () => {
    const visibleDuration = duration / zoomLevel;
    const start = timelineOffset;
    const end = Math.min(timelineOffset + visibleDuration, duration);
    return { start, end, visibleDuration };
  };

  // Get position on timeline for a time value
  const getTimelinePosition = (time: number): number => {
    const { start, visibleDuration } = getVisibleRange();
    return ((time - start) / visibleDuration) * 100;
  };

  // Zoom in/out
  const handleZoom = (delta: number) => {
    const newZoom = Math.max(1, Math.min(zoomLevel + delta, 20));
    setZoomLevel(newZoom);
    
    // Adjust offset to keep current time centered
    const visibleDuration = duration / newZoom;
    const newOffset = Math.max(0, Math.min(currentTime - visibleDuration / 2, duration - visibleDuration));
    setTimelineOffset(newOffset);
  };

  // Pan timeline
  const handlePan = (direction: number) => {
    const { visibleDuration } = getVisibleRange();
    const step = visibleDuration * 0.25;
    const newOffset = Math.max(0, Math.min(timelineOffset + step * direction, duration - visibleDuration));
    setTimelineOffset(newOffset);
  };

  // Extract frames from video at marker positions
  const extractClipsFromMarkers = async () => {
    if (!videoFile || markers.length === 0) {
      toast.error("Keine Marker gesetzt");
      return;
    }

    setIsExtracting(true);
    setExtractionProgress(0);

    try {
      const video = document.createElement("video");
      video.src = videoUrl!;
      video.crossOrigin = "anonymous";
      
      await new Promise<void>((resolve, reject) => {
        video.onloadedmetadata = () => resolve();
        video.onerror = () => reject(new Error("Video konnte nicht geladen werden"));
      });

      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d")!;

      const clips: ManualClipData[] = [];
      const totalMarkers = markers.length;
      const MAX_FRAMES_PER_CLIP = 200; // Hard limit for YOWO compatibility

      for (let markerIdx = 0; markerIdx < markers.length; markerIdx++) {
        const marker = markers[markerIdx];
        const frameTimes: number[] = [];
        
        // Calculate ALL frame times based on frameInterval (no framesPerClip limit for manual clips)
        for (let t = marker.startTime; t < marker.endTime; t += frameInterval) {
          frameTimes.push(t);
        }
        
        const allFrames: { file: File; url: string; frameNumber: number }[] = [];
        
        for (let frameIdx = 0; frameIdx < frameTimes.length; frameIdx++) {
          const time = frameTimes[frameIdx];
          
          // Seek to frame time
          video.currentTime = time;
          await new Promise<void>((resolve) => {
            video.onseeked = () => resolve();
          });
          
          // Draw frame to canvas
          ctx.drawImage(video, 0, 0);
          
          // Convert to blob
          const blob = await new Promise<Blob>((resolve) => {
            canvas.toBlob((b) => resolve(b!), "image/jpeg", 0.92);
          });
          
          const frameNumber = frameIdx + 1;
          const fileName = `${String(frameNumber).padStart(5, "0")}.jpg`;
          const file = new File([blob], fileName, { type: "image/jpeg" });
          const url = URL.createObjectURL(blob);
          
          allFrames.push({ file, url, frameNumber });
          
          // Update progress
          const overallProgress = ((markerIdx + (frameIdx + 1) / frameTimes.length) / totalMarkers) * 100;
          setExtractionProgress(Math.round(overallProgress));
        }
        
        // Split into chunks of max 200 frames if needed
        if (allFrames.length <= MAX_FRAMES_PER_CLIP) {
          // Single clip
          const clipName = `clip_${String(markerIdx + 1).padStart(3, "0")}_${formatTime(marker.startTime).replace(":", "-")}`;
          clips.push({
            name: clipName,
            frames: allFrames,
            sourceVideo: videoFile.name,
          });
        } else {
          // Split into multiple clips (by half recursively or just chunks)
          const numChunks = Math.ceil(allFrames.length / MAX_FRAMES_PER_CLIP);
          const chunkSize = Math.ceil(allFrames.length / numChunks); // Distribute evenly
          
          for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
            const start = chunkIdx * chunkSize;
            const end = Math.min(start + chunkSize, allFrames.length);
            const chunkFrames = allFrames.slice(start, end);
            
            // Renumber frames within each chunk starting from 1
            const renumberedFrames = chunkFrames.map((f, idx) => ({
              ...f,
              frameNumber: idx + 1,
            }));
            
            const clipName = `clip_${String(markerIdx + 1).padStart(3, "0")}_${formatTime(marker.startTime).replace(":", "-")}_part${chunkIdx + 1}`;
            clips.push({
              name: clipName,
              frames: renumberedFrames,
              sourceVideo: videoFile.name,
            });
          }
          
          toast.info(`Clip ${markerIdx + 1} hatte ${allFrames.length} Frames → in ${numChunks} Teile aufgeteilt`);
        }
      }

      onClipsCreated(clips);
      toast.success(`${clips.length} Clips mit ${clips.reduce((sum, c) => sum + c.frames.length, 0)} Frames erstellt!`);
      
    } catch (error) {
      console.error("Error extracting clips:", error);
      toast.error("Fehler beim Extrahieren der Clips");
    } finally {
      setIsExtracting(false);
      setExtractionProgress(0);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  // Keep current time visible in timeline
  useEffect(() => {
    if (duration === 0 || zoomLevel === 1) return;
    
    const { start, end, visibleDuration } = getVisibleRange();
    if (currentTime < start || currentTime > end) {
      const newOffset = Math.max(0, Math.min(currentTime - visibleDuration / 2, duration - visibleDuration));
      setTimelineOffset(newOffset);
    }
  }, [currentTime, duration, zoomLevel]);

  // Cleanup frame stepping on unmount
  useEffect(() => {
    return () => {
      if (frameRequestRef.current) {
        cancelAnimationFrame(frameRequestRef.current);
      }
    };
  }, []);

  // Handle space key for marker creation - USES videoRef.current.currentTime directly for accuracy!
  const handleSpaceMarker = useCallback(() => {
    if (duration === 0 || !videoRef.current) return;
    
    // CRITICAL: Use the ACTUAL video time, not the React state (which may be stale after frame stepping)
    const actualVideoTime = videoRef.current.currentTime;
    const snappedTime = snapToGrid(actualVideoTime);
    
    if (pendingMarkerStart === null) {
      // First press - set start
      setPendingMarkerStart(snappedTime);
      // Also sync the React state to match
      setCurrentTime(actualVideoTime);
      toast.info(`▶ Start: ${formatTime(snappedTime)} - Drücke Space für Ende`);
    } else {
      // Second press - create marker
      const startTime = Math.min(pendingMarkerStart, snappedTime);
      const endTime = Math.max(pendingMarkerStart, snappedTime);
      
      if (endTime - startTime < 0.5) {
        toast.error("Clip muss mindestens 0.5 Sekunden lang sein");
        setPendingMarkerStart(null);
        return;
      }
      
      const newMarker: ClipMarker = {
        id: Date.now().toString(),
        startTime,
        endTime,
      };
      
      setMarkers(prev => [...prev, newMarker].sort((a, b) => a.startTime - b.startTime));
      setPendingMarkerStart(null);
      // Sync state
      setCurrentTime(actualVideoTime);
      toast.success(`✓ Clip erstellt: ${formatTime(startTime)} - ${formatTime(endTime)}`);
    }
  }, [duration, pendingMarkerStart]);

  // Keyboard shortcuts for frame stepping and marker creation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!videoUrl) return;
      
      // Skip if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      
      // Ignore key repeats for arrow keys - we handle continuous stepping ourselves
      if (e.repeat && (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === ',' || e.key === '.')) {
        e.preventDefault();
        return;
      }
      
      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault();
          startContinuousStep(-1);
          break;
        case 'ArrowRight':
          e.preventDefault();
          startContinuousStep(1);
          break;
        case ' ':
          e.preventDefault();
          // Space = Set marker (start or end)
          handleSpaceMarker();
          break;
        case ',':
          e.preventDefault();
          startContinuousStep(-1);
          break;
        case '.':
          e.preventDefault();
          startContinuousStep(1);
          break;
        case 'p':
        case 'P':
          e.preventDefault();
          togglePlayPause();
          break;
        case 'Escape':
          if (pendingMarkerStart !== null) {
            e.preventDefault();
            cancelPendingMarker();
          }
          break;
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      // Stop continuous stepping when key is released
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === ',' || e.key === '.') {
        stopContinuousStep();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [videoUrl, isPlaying, duration, pendingMarkerStart, startContinuousStep, stopContinuousStep, handleSpaceMarker]);

  const { start: visibleStart, end: visibleEnd } = getVisibleRange();

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Scissors className="w-5 h-5" />
          Manueller Clip-Editor
        </h3>
        <input
          ref={videoInputRef}
          type="file"
          accept="video/*"
          onChange={handleVideoUpload}
          className="hidden"
        />
        <Button
          onClick={() => videoInputRef.current?.click()}
          variant="outline"
          size="sm"
        >
          <Video className="w-4 h-4 mr-2" />
          Video laden
        </Button>
      </div>

      {/* Detection Settings - Always visible */}
      <div className="p-4 bg-secondary/50 rounded-lg space-y-4 mb-4">
        <h4 className="text-sm font-semibold flex items-center gap-2">
          Erkennungs-Einstellungen
          <span className="text-xs text-muted-foreground font-normal">(werden beim Annotieren angewendet)</span>
        </h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Confidence Threshold */}
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-muted-foreground">Konfidenz-Schwellenwert</span>
              <span className="font-bold">{(confidenceThreshold * 100).toFixed(0)}%</span>
            </div>
            <Slider
              value={[confidenceThreshold]}
              onValueChange={([v]) => setConfidenceThreshold(v)}
              min={0.1}
              max={0.95}
              step={0.05}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Niedrigerer Wert = mehr Erkennungen
            </p>
          </div>
          
          {/* Bounding Box Padding */}
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-muted-foreground">Bounding Box Padding</span>
              <span className="font-bold">{boundingBoxPadding}%</span>
            </div>
            <Slider
              value={[boundingBoxPadding]}
              onValueChange={([v]) => setBoundingBoxPadding(v)}
              min={0}
              max={50}
              step={5}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Erweitert Boxen für Werkzeuge
            </p>
          </div>
        </div>
        
        {/* Motion Blur Filter */}
        <div className="pt-2 border-t border-border/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-muted-foreground">Bewegungsunschärfe filtern</span>
            <Button
              variant={filterMotionBlur ? "default" : "outline"}
              size="sm"
              onClick={() => setFilterMotionBlur(!filterMotionBlur)}
            >
              {filterMotionBlur ? "Aktiv" : "Aus"}
            </Button>
          </div>
          {filterMotionBlur && (
            <div className="mt-2">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-muted-foreground">Blur-Schwellenwert</span>
                <span className="font-bold">{blurThreshold}</span>
              </div>
              <Slider
                value={[blurThreshold]}
                onValueChange={([v]) => setBlurThreshold(v)}
                min={50}
                max={300}
                step={10}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Höher = weniger streng
              </p>
            </div>
          )}
        </div>

        {/* Video Enhancement / Upscaling */}
        <div className="pt-2 border-t border-border/50">
          <div className="flex items-center justify-between mb-2">
            <div>
              <span className="text-sm text-muted-foreground">Video-Verbesserung (Upscaling)</span>
              <p className="text-xs text-muted-foreground">Verbessert schlechte Videoqualität für bessere Erkennung</p>
            </div>
            <Button
              variant={enhanceVideo ? "default" : "outline"}
              size="sm"
              onClick={() => setEnhanceVideo(!enhanceVideo)}
            >
              {enhanceVideo ? "Aktiv" : "Aus"}
            </Button>
          </div>
          {enhanceVideo && (
            <div className="mt-2">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-muted-foreground">Skalierungsfaktor</span>
                <span className="font-bold">{enhanceScale}x</span>
              </div>
              <Slider
                value={[enhanceScale]}
                onValueChange={([v]) => setEnhanceScale(v)}
                min={1.5}
                max={4}
                step={0.5}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Höher = bessere Details, aber langsamer. 2x empfohlen.
              </p>
            </div>
          )}
        </div>
      </div>

      {!videoUrl ? (
        <div 
          className="border-2 border-dashed rounded-lg p-12 text-center cursor-pointer hover:border-primary/50 transition-colors"
          onClick={() => videoInputRef.current?.click()}
        >
          <Video className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground">Klicken um Video zu laden</p>
          <p className="text-sm text-muted-foreground mt-2">
            Dann Clip-Marker setzen und Frames extrahieren
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Video Player */}
          <div className="relative bg-black rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              src={videoUrl}
              className="w-full max-h-[400px] object-contain"
              onLoadedMetadata={handleVideoLoaded}
              onTimeUpdate={handleTimeUpdate}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onEnded={() => setIsPlaying(false)}
            />
          </div>

          {/* Playback Controls */}
          <div className="flex items-center gap-2">
            {/* Frame back */}
            <Button 
              onClick={() => stepFrameInstant(-1)}
              variant="outline" 
              size="sm"
              title="1 Frame zurück (← oder ,)"
            >
              <SkipBack className="w-4 h-4" />
            </Button>
            
            {/* Play/Pause */}
            <Button onClick={togglePlayPause} variant="outline" size="sm" title="Play/Pause (P)">
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </Button>
            
            {/* Playback Speed Toggle */}
            <Button 
              onClick={togglePlaybackSpeed} 
              variant={playbackSpeed === 2 ? "default" : "outline"} 
              size="sm"
              title="Wiedergabegeschwindigkeit umschalten"
              className="font-mono min-w-[40px]"
            >
              {playbackSpeed}x
            </Button>
            
            {/* Frame forward */}
            <Button 
              onClick={() => stepFrameInstant(1)}
              variant="outline" 
              size="sm"
              title="1 Frame vor (→ oder .)"
            >
              <SkipForward className="w-4 h-4" />
            </Button>
            
            {/* Slider */}
            <div className="flex-1">
              <Slider
                value={[currentTime]}
                onValueChange={([v]) => seekTo(v)}
                min={0}
                max={duration || 1}
                step={0.001}
                className="cursor-pointer"
              />
            </div>
            
            {/* Time display with frame info */}
            <span className="text-sm text-muted-foreground font-mono min-w-[140px] text-right">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>
          
          {/* Keyboard shortcuts hint */}
          <div className="flex items-center gap-4 text-xs text-muted-foreground flex-wrap">
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-primary/20 text-primary rounded text-[10px] font-mono font-bold">Space</kbd>
              Marker setzen
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-muted rounded text-[10px] font-mono">←</kbd>
              <kbd className="px-1.5 py-0.5 bg-muted rounded text-[10px] font-mono">→</kbd>
              Frame
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-muted rounded text-[10px] font-mono">P</kbd>
              Play/Pause
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-muted rounded text-[10px] font-mono">Esc</kbd>
              Abbrechen
            </span>
          </div>

          {/* Timeline with Markers */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Timeline</span>
              <div className="flex items-center gap-2">
                <Button onClick={() => handlePan(-1)} variant="ghost" size="sm" disabled={zoomLevel === 1}>
                  <ChevronLeft className="w-4 h-4" />
                </Button>
                <Button onClick={() => handleZoom(-0.5)} variant="ghost" size="sm" disabled={zoomLevel <= 1}>
                  <ZoomOut className="w-4 h-4" />
                </Button>
                <span className="text-xs text-muted-foreground w-12 text-center">{zoomLevel.toFixed(1)}x</span>
                <Button onClick={() => handleZoom(0.5)} variant="ghost" size="sm">
                  <ZoomIn className="w-4 h-4" />
                </Button>
                <Button onClick={() => handlePan(1)} variant="ghost" size="sm" disabled={zoomLevel === 1}>
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            </div>
            
            <div
              ref={timelineRef}
              className={`relative h-32 rounded-lg cursor-crosshair overflow-hidden border ${pendingMarkerStart !== null ? 'bg-primary/10 ring-2 ring-primary border-primary' : 'bg-background border-border'}`}
              onClick={handleTimelineClick}
            >
              {/* Grid background with second markers */}
              <div className="absolute inset-0">
                {/* Generate grid lines for each second */}
                {(() => {
                  const visibleDuration = visibleEnd - visibleStart;
                  const lines: JSX.Element[] = [];
                  
                  // Determine appropriate interval based on zoom
                  let interval = 1; // 1 second default
                  if (visibleDuration > 60) interval = 10;
                  else if (visibleDuration > 30) interval = 5;
                  else if (visibleDuration > 10) interval = 2;
                  else if (visibleDuration < 5) interval = 0.5;
                  
                  // Find first line position
                  const firstLine = Math.ceil(visibleStart / interval) * interval;
                  
                  for (let time = firstLine; time <= visibleEnd; time += interval) {
                    const position = getTimelinePosition(time);
                    if (position < 0 || position > 100) continue;
                    
                    const isMainSecond = time % 1 === 0;
                    const isFiveSecond = time % 5 === 0;
                    const isTenSecond = time % 10 === 0;
                    
                    lines.push(
                      <div
                        key={`grid-${time}`}
                        className="absolute top-0 bottom-0 pointer-events-none"
                        style={{ left: `${position}%` }}
                      >
                        {/* Grid line */}
                        <div 
                          className={`w-px h-full ${
                            isTenSecond 
                              ? 'bg-foreground/30' 
                              : isFiveSecond 
                                ? 'bg-foreground/20' 
                                : isMainSecond 
                                  ? 'bg-foreground/10' 
                                  : 'bg-foreground/5'
                          }`} 
                        />
                      </div>
                    );
                  }
                  return lines;
                })()}
                
                {/* Sub-grid for half seconds when zoomed in */}
                {(() => {
                  const visibleDuration = visibleEnd - visibleStart;
                  if (visibleDuration > 10) return null;
                  
                  const lines: JSX.Element[] = [];
                  const interval = 0.5;
                  const firstLine = Math.ceil(visibleStart / interval) * interval;
                  
                  for (let time = firstLine; time <= visibleEnd; time += interval) {
                    if (time % 1 === 0) continue; // Skip full seconds
                    const position = getTimelinePosition(time);
                    if (position < 0 || position > 100) continue;
                    
                    lines.push(
                      <div
                        key={`subgrid-${time}`}
                        className="absolute top-0 bottom-0 pointer-events-none"
                        style={{ left: `${position}%` }}
                      >
                        <div className="w-px h-full bg-foreground/5" />
                      </div>
                    );
                  }
                  return lines;
                })()}
              </div>

              {/* Time labels at bottom */}
              <div className="absolute bottom-0 left-0 right-0 h-6 bg-muted/50 border-t border-border">
                {(() => {
                  const visibleDuration = visibleEnd - visibleStart;
                  const labels: JSX.Element[] = [];
                  
                  // Determine label interval
                  let labelInterval = 1;
                  if (visibleDuration > 120) labelInterval = 30;
                  else if (visibleDuration > 60) labelInterval = 10;
                  else if (visibleDuration > 30) labelInterval = 5;
                  else if (visibleDuration > 10) labelInterval = 2;
                  else if (visibleDuration < 3) labelInterval = 0.5;
                  
                  const firstLabel = Math.ceil(visibleStart / labelInterval) * labelInterval;
                  
                  for (let time = firstLabel; time <= visibleEnd; time += labelInterval) {
                    const position = getTimelinePosition(time);
                    if (position < 2 || position > 98) continue;
                    
                    labels.push(
                      <div
                        key={`label-${time}`}
                        className="absolute bottom-0 h-full flex flex-col items-center justify-center transform -translate-x-1/2"
                        style={{ left: `${position}%` }}
                      >
                        <div className="h-2 w-px bg-foreground/40 mb-0.5" />
                        <span className="text-[9px] font-mono text-muted-foreground leading-none">
                          {formatTimeShort(time)}
                        </span>
                      </div>
                    );
                  }
                  return labels;
                })()}
              </div>

              {/* Clip markers area */}
              <div className="absolute top-0 left-0 right-0 bottom-6">
                {/* Pending marker start indicator */}
                {pendingMarkerStart !== null && pendingMarkerStart >= visibleStart && pendingMarkerStart <= visibleEnd && (
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-primary z-20"
                    style={{ left: `${getTimelinePosition(pendingMarkerStart)}%` }}
                  >
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 w-5 h-5 bg-primary rounded-b-lg flex items-center justify-center shadow-lg">
                      <span className="text-[10px] text-primary-foreground font-bold">S</span>
                    </div>
                    {/* Dotted line showing potential range */}
                    <div className="absolute top-5 bottom-0 left-1/2 -translate-x-1/2 w-px border-l border-dashed border-primary" />
                  </div>
                )}

                {/* Clip markers */}
                {markers.map((marker, idx) => {
                  const leftPos = getTimelinePosition(marker.startTime);
                  const rightPos = getTimelinePosition(marker.endTime);
                  const width = rightPos - leftPos;
                  
                  // Only show if visible
                  if (rightPos < 0 || leftPos > 100) return null;
                  
                  return (
                    <div
                      key={marker.id}
                      className="absolute top-2 bottom-2 bg-primary/20 border-2 border-primary rounded-lg cursor-pointer hover:bg-primary/30 transition-colors group"
                      style={{
                        left: `${Math.max(0, leftPos)}%`,
                        width: `${Math.min(100 - Math.max(0, leftPos), width)}%`,
                        minWidth: "24px",
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        seekTo(marker.startTime);
                      }}
                    >
                      {/* Clip label */}
                      <div className="absolute inset-0 flex items-center justify-center overflow-hidden">
                        <span className="text-[10px] text-primary font-semibold bg-background/80 px-1.5 py-0.5 rounded whitespace-nowrap">
                          Clip {idx + 1}
                        </span>
                      </div>
                      {/* Duration badge */}
                      <div className="absolute -top-1 -right-1 bg-primary text-primary-foreground text-[8px] px-1 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                        {(marker.endTime - marker.startTime).toFixed(1)}s
                      </div>
                      {/* Resize handles */}
                      <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary rounded-l-lg cursor-ew-resize hover:bg-primary/80" />
                      <div className="absolute right-0 top-0 bottom-0 w-1 bg-primary rounded-r-lg cursor-ew-resize hover:bg-primary/80" />
                    </div>
                  );
                })}

                {/* Current time indicator (playhead) */}
                {currentTime >= visibleStart && currentTime <= visibleEnd && (
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-destructive z-30 pointer-events-none"
                    style={{ left: `${getTimelinePosition(currentTime)}%` }}
                  >
                    <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-0 h-0 border-l-[6px] border-r-[6px] border-t-[8px] border-l-transparent border-r-transparent border-t-destructive" />
                    <div className="absolute top-2 left-1/2 -translate-x-1/2 bg-destructive text-destructive-foreground text-[8px] px-1 py-0.5 rounded font-mono whitespace-nowrap">
                      {formatTimeShort(currentTime)}
                    </div>
                  </div>
                )}
              </div>
              
              {/* Snap indicator overlay */}
              <div className="absolute top-1 right-1 bg-background/80 text-[9px] text-muted-foreground px-1.5 py-0.5 rounded flex items-center gap-1 pointer-events-none">
                <div className="w-1.5 h-1.5 bg-primary rounded-full" />
                Snap: 0.5s
              </div>
            </div>
          </div>

          {/* Instructions & Actions */}
          <div className="flex gap-3 items-center">
            <div className="flex-1 text-sm text-muted-foreground">
              {pendingMarkerStart !== null ? (
                <span className="text-primary font-medium">
                  ▶ Start bei {formatTime(pendingMarkerStart)} - Klicke auf Timeline für Ende
                </span>
              ) : (
                <span>Klicke auf Timeline: 1. Klick = Start, 2. Klick = Ende</span>
              )}
            </div>
            
            {pendingMarkerStart !== null && (
              <Button onClick={cancelPendingMarker} variant="outline" size="sm">
                <X className="w-4 h-4 mr-1" />
                Abbrechen
              </Button>
            )}
            
            {markers.length > 0 && (
              <Button
                onClick={extractClipsFromMarkers}
                disabled={isExtracting || pendingMarkerStart !== null}
              >
                {isExtracting ? (
                  <>Extrahiere... {extractionProgress}%</>
                ) : (
                  <>
                    <Check className="w-4 h-4 mr-2" />
                    {markers.length} Clips erstellen
                  </>
                )}
              </Button>
            )}
          </div>

          {/* Markers List */}
          {markers.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium">Clip-Marker ({markers.length})</h4>
              <div className="max-h-48 overflow-y-auto space-y-2">
                {markers.map((marker, idx) => (
                  <div
                    key={marker.id}
                    className="flex items-center gap-3 p-2 bg-secondary rounded-lg text-sm"
                  >
                    <span className="font-medium text-primary min-w-[60px]">Clip {idx + 1}</span>
                    
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground">Start:</span>
                      <input
                        type="number"
                        value={marker.startTime.toFixed(1)}
                        onChange={(e) => updateMarker(marker.id, "startTime", parseFloat(e.target.value) || 0)}
                        className="w-16 h-6 px-1 rounded border border-input bg-background text-xs"
                        step="0.5"
                      />
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground">Ende:</span>
                      <input
                        type="number"
                        value={marker.endTime.toFixed(1)}
                        onChange={(e) => updateMarker(marker.id, "endTime", parseFloat(e.target.value) || 0)}
                        className="w-16 h-6 px-1 rounded border border-input bg-background text-xs"
                        step="0.5"
                      />
                    </div>
                    
                    <span className="text-muted-foreground ml-auto">
                      ~{Math.floor((marker.endTime - marker.startTime) / frameInterval)} Frames
                    </span>
                    
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0"
                      onClick={() => seekTo(marker.startTime)}
                    >
                      <Play className="w-3 h-3" />
                    </Button>
                    
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0 text-destructive hover:text-destructive"
                      onClick={() => deleteMarker(marker.id)}
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Progress */}
          {isExtracting && (
            <div className="w-full bg-secondary rounded-full h-2">
              <div
                className="bg-primary h-2 rounded-full transition-all duration-300"
                style={{ width: `${extractionProgress}%` }}
              />
            </div>
          )}
        </div>
      )}
    </Card>
  );
};
