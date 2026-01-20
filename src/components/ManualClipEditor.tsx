import { useState, useRef, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, Trash2, Scissors, Check, Video, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, X } from "lucide-react";
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
  onClipsCreated: (clips: ManualClipData[]) => void;
}

export const ManualClipEditor = ({ framesPerClip, frameInterval, onClipsCreated }: ManualClipEditorProps) => {
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
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  // Format time as MM:SS.ss
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toFixed(1).padStart(4, "0")}`;
  };

  // Handle timeline click - TWO CLICK MARKER CREATION
  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!timelineRef.current || duration === 0) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = x / rect.width;
    
    // Calculate time based on zoom and offset
    const visibleDuration = duration / zoomLevel;
    const clickedTime = Math.max(0, Math.min(timelineOffset + ratio * visibleDuration, duration));
    
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

      for (let markerIdx = 0; markerIdx < markers.length; markerIdx++) {
        const marker = markers[markerIdx];
        const clipDuration = marker.endTime - marker.startTime;
        const frameTimes: number[] = [];
        
        // Calculate frame times based on frameInterval
        for (let t = marker.startTime; t < marker.endTime; t += frameInterval) {
          frameTimes.push(t);
        }
        
        // Limit to framesPerClip
        const limitedFrameTimes = frameTimes.slice(0, framesPerClip);
        
        const frames: { file: File; url: string; frameNumber: number }[] = [];
        
        for (let frameIdx = 0; frameIdx < limitedFrameTimes.length; frameIdx++) {
          const time = limitedFrameTimes[frameIdx];
          
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
          
          frames.push({ file, url, frameNumber });
          
          // Update progress
          const overallProgress = ((markerIdx + (frameIdx + 1) / limitedFrameTimes.length) / totalMarkers) * 100;
          setExtractionProgress(Math.round(overallProgress));
        }
        
        const clipName = `clip_${String(markerIdx + 1).padStart(3, "0")}_${formatTime(marker.startTime).replace(":", "-")}`;
        clips.push({
          name: clipName,
          frames,
          sourceVideo: videoFile.name,
        });
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
          <div className="flex items-center gap-4">
            <Button onClick={togglePlayPause} variant="outline" size="sm">
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </Button>
            
            <div className="flex-1">
              <Slider
                value={[currentTime]}
                onValueChange={([v]) => seekTo(v)}
                min={0}
                max={duration || 1}
                step={0.1}
              />
            </div>
            
            <span className="text-sm text-muted-foreground font-mono min-w-[100px]">
              {formatTime(currentTime)} / {formatTime(duration)}
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
              className={`relative h-20 rounded-lg cursor-crosshair overflow-hidden ${pendingMarkerStart !== null ? 'bg-primary/20 ring-2 ring-primary' : 'bg-secondary'}`}
              onClick={handleTimelineClick}
            >
              {/* Time markers */}
              <div className="absolute inset-0 flex items-end pb-1">
                {Array.from({ length: 11 }).map((_, i) => {
                  const time = visibleStart + (visibleEnd - visibleStart) * (i / 10);
                  return (
                    <div
                      key={i}
                      className="absolute bottom-0 text-[10px] text-muted-foreground transform -translate-x-1/2"
                      style={{ left: `${i * 10}%` }}
                    >
                      <div className="h-2 w-px bg-muted-foreground/30 mx-auto mb-0.5" />
                      {formatTime(time)}
                    </div>
                  );
                })}
              </div>

              {/* Pending marker start indicator */}
              {pendingMarkerStart !== null && pendingMarkerStart >= visibleStart && pendingMarkerStart <= visibleEnd && (
                <div
                  className="absolute top-0 bottom-4 w-1 bg-primary z-20 animate-pulse"
                  style={{ left: `${getTimelinePosition(pendingMarkerStart)}%` }}
                >
                  <div className="absolute -top-0 left-1/2 -translate-x-1/2 w-4 h-4 bg-primary rounded-full flex items-center justify-center">
                    <span className="text-[8px] text-primary-foreground font-bold">S</span>
                  </div>
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
                    className="absolute top-2 h-8 bg-primary/30 border-2 border-primary rounded cursor-pointer hover:bg-primary/40 transition-colors"
                    style={{
                      left: `${Math.max(0, leftPos)}%`,
                      width: `${Math.min(100 - Math.max(0, leftPos), width)}%`,
                      minWidth: "20px",
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      seekTo(marker.startTime);
                    }}
                  >
                    <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-[10px] text-primary font-medium whitespace-nowrap">
                      Clip {idx + 1}
                    </span>
                  </div>
                );
              })}

              {/* Current time indicator */}
              {currentTime >= visibleStart && currentTime <= visibleEnd && (
                <div
                  className="absolute top-0 bottom-4 w-0.5 bg-destructive z-10"
                  style={{ left: `${getTimelinePosition(currentTime)}%` }}
                >
                  <div className="absolute -top-0 left-1/2 -translate-x-1/2 w-3 h-3 bg-destructive rounded-full" />
                </div>
              )}
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
