import { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Upload, ChevronLeft, ChevronRight, Download, Trash2, Tag, Sparkles, SlidersHorizontal, Video, Film, Pencil, Check, X, Loader2 } from "lucide-react";
import { toast } from "sonner";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";
import JSZip from "jszip";


interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  confidence: number;
}

interface ImageData {
  file: File;
  url: string;
  detections: Detection[];
  frameNumber?: number;
  sourceVideo?: string;
}

type ResizeHandle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w" | "move" | null;

export const ImageAnnotator = () => {
  const [images, setImages] = useState<ImageData[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [selectedDetection, setSelectedDetection] = useState<number | null>(null);
  const [resizing, setResizing] = useState<{
    handle: ResizeHandle;
    startX: number;
    startY: number;
    originalDetection: Detection;
  } | null>(null);
  const [previewDetection, setPreviewDetection] = useState<Detection | null>(null);
  const [customLabel, setCustomLabel] = useState<string>("");
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [detectAllClasses, setDetectAllClasses] = useState(true);
  const [frameInterval, setFrameInterval] = useState(1); // Extract every N seconds
  const [extractionProgress, setExtractionProgress] = useState(0);
  const [editingLabel, setEditingLabel] = useState<number | null>(null);
  const [editLabelValue, setEditLabelValue] = useState("");
  const [boundingBoxPadding, setBoundingBoxPadding] = useState(15); // Padding in %
  const [customActivityLabels, setCustomActivityLabels] = useState<string[]>([]);
  const [newCustomLabel, setNewCustomLabel] = useState("");
  
  
  // Preset activity labels for quick selection - Schweißumfeld Tätigkeiten
  const defaultActivityLabels = [
    "Transport", 
    "MAG-Schweißen", 
    "Putzen/Nacharbeiten", 
    "Zwischenkontrolle"
  ];
  
  // Kombiniere Standard-Labels mit benutzerdefinierten Labels
  const activityLabels = [...defaultActivityLabels, ...customActivityLabels];
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const [scale, setScale] = useState(1);

  const currentImage = images[currentIndex];

  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log("Initializing TensorFlow.js with WebGPU...");
        
        let backend = "webgl";
        try {
          await tf.setBackend("webgpu");
          await tf.ready();
          backend = "webgpu";
          console.log("✅ WebGPU backend activated (2-3x faster!)");
          toast.success("WebGPU aktiviert - maximale Performance! 🚀");
        } catch (webgpuError) {
          console.warn("WebGPU not available, falling back to WebGL:", webgpuError);
          await tf.setBackend("webgl");
          await tf.ready();
          toast.success("WebGL aktiviert");
        }
        
        console.log(`Loading COCO-SSD model on ${backend}...`);
        const loadedModel = await cocoSsd.load();
        setModel(loadedModel);
        console.log("Model loaded successfully!");
        
        if ("caches" in window) {
          const cache = await caches.open("tfjs-models-cache");
          const cachedModel = await cache.match("https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/model.json");
          if (cachedModel) {
            toast.success("✅ Offline-Modus aktiv - keine Internetverbindung nötig!", { duration: 4000 });
          }
        }
      } catch (error) {
        console.error("Error loading model:", error);
        toast.error("Fehler beim Laden des KI-Modells");
      }
    };
    loadModel();
  }, []);

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === "d" || e.key === "D") {
        handleNext();
      } else if (e.key === "a" || e.key === "A") {
        handlePrevious();
      } else if (e.key === "Delete" || e.key === "Backspace") {
        if (selectedDetection !== null) {
          handleDeleteDetection(selectedDetection);
        }
      } else if (e.key === "Escape") {
        setSelectedDetection(null);
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [currentIndex, images.length, selectedDetection]);

  // Calculate scale when image loads
  const updateScale = useCallback(() => {
    if (!currentImage || !canvasRef.current || !containerRef.current) return;
    
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const displayWidth = canvas.getBoundingClientRect().width;
    const actualWidth = canvas.width;
    
    if (actualWidth > 0) {
      setScale(displayWidth / actualWidth);
    }
  }, [currentImage]);

  useEffect(() => {
    updateScale();
    window.addEventListener("resize", updateScale);
    return () => window.removeEventListener("resize", updateScale);
  }, [updateScale]);

  // Draw image and detections on canvas - use requestAnimationFrame for smooth updates
  const drawCanvas = useCallback(() => {
    if (!currentImage || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.src = currentImage.url;
    
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      
      ctx.drawImage(img, 0, 0);
      
      currentImage.detections.forEach((detection, idx) => {
        const isSelected = idx === selectedDetection;
        // Use preview detection if we're resizing this one
        const drawDetection = (isSelected && previewDetection) ? previewDetection : detection;
        
        // Draw bounding box
        ctx.strokeStyle = isSelected ? "hsl(45, 100%, 50%)" : "hsl(189, 90%, 55%)";
        ctx.lineWidth = isSelected ? 4 : 3;
        ctx.strokeRect(drawDetection.x, drawDetection.y, drawDetection.width, drawDetection.height);
        
        // Draw label background
        ctx.fillStyle = isSelected ? "hsl(45, 100%, 50%)" : "hsl(189, 90%, 55%)";
        const labelText = `${drawDetection.label} ${(drawDetection.confidence * 100).toFixed(0)}%`;
        const metrics = ctx.measureText(labelText);
        ctx.fillRect(drawDetection.x, drawDetection.y - 25, metrics.width + 12, 25);
        
        // Draw label text
        ctx.fillStyle = "hsl(220, 25%, 10%)";
        ctx.font = "14px system-ui, -apple-system, sans-serif";
        ctx.fillText(labelText, drawDetection.x + 6, drawDetection.y - 8);
        
        // Draw resize handles for selected detection - ONLY CORNERS, much bigger
        if (isSelected) {
          const handleSize = 20;
          
          // Corner handles only - big and easy to grab
          const corners = [
            { x: drawDetection.x, y: drawDetection.y }, // NW
            { x: drawDetection.x + drawDetection.width, y: drawDetection.y }, // NE
            { x: drawDetection.x + drawDetection.width, y: drawDetection.y + drawDetection.height }, // SE
            { x: drawDetection.x, y: drawDetection.y + drawDetection.height }, // SW
          ];
          
          corners.forEach(corner => {
            // White fill with yellow border for visibility
            ctx.fillStyle = "white";
            ctx.strokeStyle = "hsl(45, 100%, 50%)";
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(corner.x, corner.y, handleSize / 2, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
          });
        }
      });
      
      // Update scale after drawing
      setTimeout(updateScale, 10);
    };
  }, [currentImage, selectedDetection, previewDetection, updateScale]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  const getHandleAtPosition = (x: number, y: number, detection: Detection): ResizeHandle => {
    const handleRadius = 25; // Big hit area for easy grabbing
    const { x: dx, y: dy, width, height } = detection;
    
    // Check corners only - with distance calculation
    const distNW = Math.sqrt(Math.pow(x - dx, 2) + Math.pow(y - dy, 2));
    const distNE = Math.sqrt(Math.pow(x - (dx + width), 2) + Math.pow(y - dy, 2));
    const distSE = Math.sqrt(Math.pow(x - (dx + width), 2) + Math.pow(y - (dy + height), 2));
    const distSW = Math.sqrt(Math.pow(x - dx, 2) + Math.pow(y - (dy + height), 2));
    
    if (distNW < handleRadius) return "nw";
    if (distNE < handleRadius) return "ne";
    if (distSE < handleRadius) return "se";
    if (distSW < handleRadius) return "sw";
    
    // Check if inside box for move
    if (x >= dx && x <= dx + width && y >= dy && y <= dy + height) return "move";
    
    return null;
  };

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!currentImage || !canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / scale;
    const y = (e.clientY - rect.top) / scale;
    
    // Check if clicking on a selected detection's handle
    if (selectedDetection !== null) {
      const detection = currentImage.detections[selectedDetection];
      const handle = getHandleAtPosition(x, y, detection);
      
      if (handle) {
        setResizing({
          handle,
          startX: x,
          startY: y,
          originalDetection: { ...detection },
        });
        return;
      }
    }
    
    // Check if clicking on any detection
    for (let i = currentImage.detections.length - 1; i >= 0; i--) {
      const detection = currentImage.detections[i];
      if (
        x >= detection.x &&
        x <= detection.x + detection.width &&
        y >= detection.y &&
        y <= detection.y + detection.height
      ) {
        setSelectedDetection(i);
        setResizing({
          handle: "move",
          startX: x,
          startY: y,
          originalDetection: { ...detection },
        });
        return;
      }
    }
    
    // Clicked outside all detections
    setSelectedDetection(null);
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!resizing || !currentImage || selectedDetection === null || !canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / scale;
    const y = (e.clientY - rect.top) / scale;
    
    const dx = x - resizing.startX;
    const dy = y - resizing.startY;
    const orig = resizing.originalDetection;
    
    let newDetection = { ...orig };
    
    switch (resizing.handle) {
      case "move":
        newDetection.x = Math.max(0, orig.x + dx);
        newDetection.y = Math.max(0, orig.y + dy);
        break;
      case "nw":
        newDetection.x = Math.max(0, orig.x + dx);
        newDetection.y = Math.max(0, orig.y + dy);
        newDetection.width = orig.width - dx;
        newDetection.height = orig.height - dy;
        break;
      case "ne":
        newDetection.y = Math.max(0, orig.y + dy);
        newDetection.width = orig.width + dx;
        newDetection.height = orig.height - dy;
        break;
      case "se":
        newDetection.width = orig.width + dx;
        newDetection.height = orig.height + dy;
        break;
      case "sw":
        newDetection.x = Math.max(0, orig.x + dx);
        newDetection.width = orig.width - dx;
        newDetection.height = orig.height + dy;
        break;
    }
    
    // Ensure minimum size
    if (newDetection.width < 20) newDetection.width = 20;
    if (newDetection.height < 20) newDetection.height = 20;
    
    // Update preview only - not the actual state (real-time update)
    setPreviewDetection(newDetection);
  };

  const handleCanvasMouseUp = () => {
    // Commit the preview to actual state
    if (resizing && previewDetection && selectedDetection !== null) {
      const newImages = [...images];
      newImages[currentIndex].detections[selectedDetection] = previewDetection;
      setImages(newImages);
    }
    setResizing(null);
    setPreviewDetection(null);
  };

  const getCursorStyle = (): string => {
    if (!currentImage || !canvasRef.current) return "default";
    
    if (resizing) {
      switch (resizing.handle) {
        case "nw":
        case "se":
          return "nwse-resize";
        case "ne":
        case "sw":
          return "nesw-resize";
        case "n":
        case "s":
          return "ns-resize";
        case "e":
        case "w":
          return "ew-resize";
        case "move":
          return "move";
        default:
          return "default";
      }
    }
    
    return selectedDetection !== null ? "move" : "default";
  };

  const handleDeleteDetection = (idx: number) => {
    const newImages = [...images];
    newImages[currentIndex].detections.splice(idx, 1);
    setImages(newImages);
    setSelectedDetection(null);
    toast.success("Erkennung gelöscht");
  };

  const detectObjects = async (imageFile: File, threshold: number = 0.5): Promise<Detection[]> => {
    if (!model) {
      console.log("Model not loaded yet");
      return [];
    }

    return new Promise((resolve) => {
      const img = new Image();
      img.src = URL.createObjectURL(imageFile);
      
      img.onload = async () => {
        try {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          if (!ctx) {
            resolve([]);
            return;
          }

          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);

          const predictions = await model.detect(canvas);
          console.log("Raw predictions:", predictions);
          
          // Detect all classes or filter to person only
          const detections = predictions
            .filter(pred => pred.score >= threshold)
            .map(pred => {
              // Erweitere Bounding Box um einstellbaren Prozentsatz links/rechts für Werkzeuge in der Hand
              const padding = pred.bbox[2] * (boundingBoxPadding / 100);
              
              const x = Math.max(0, pred.bbox[0] - padding);
              const y = Math.max(0, pred.bbox[1]);
              const width = Math.min(pred.bbox[2] + padding * 2, img.width - x);
              const height = Math.min(pred.bbox[3], img.height - y);
              
              // Capitalize first letter of class name
              const label = pred.class.charAt(0).toUpperCase() + pred.class.slice(1);
              
              return {
                x,
                y,
                width,
                height,
                label,
                confidence: pred.score,
              };
            })
            .filter(det => det.width > 0 && det.height > 0);
          
          console.log("Valid detections:", detections);
          resolve(detections);
        } catch (error) {
          console.error("Detection error:", error);
          resolve([]);
        }
      };
      
      img.onerror = () => {
        console.error("Image load error");
        resolve([]);
      };
    });
  };

  // Constants for video chunking
  const CHUNK_DURATION = 300; // 5 minutes in seconds

  // Extract frames from a video chunk (startTime to endTime)
  const extractFramesFromChunk = async (
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D,
    videoFile: File,
    startTime: number,
    endTime: number,
    globalFrameOffset: number
  ): Promise<ImageData[]> => {
    const frames: ImageData[] = [];
    const chunkFrames = Math.floor((endTime - startTime) / frameInterval);
    let skippedFrames = 0;
    
    for (let i = 0; i <= chunkFrames; i++) {
      const currentTime = startTime + (i * frameInterval);
      if (currentTime > endTime) break;
      
      video.currentTime = currentTime;
      
      await new Promise<void>((seekResolve) => {
        video.onseeked = () => seekResolve();
      });
      
      ctx.drawImage(video, 0, 0);
      
      // Convert canvas to blob
      const blob = await new Promise<Blob>((blobResolve) => {
        canvas.toBlob((b) => blobResolve(b!), 'image/jpeg', 0.95);
      });
      
      const frameNumber = globalFrameOffset + i;
      const frameFile = new File([blob], `${videoFile.name}_frame_${frameNumber.toString().padStart(4, '0')}.jpg`, { type: 'image/jpeg' });
      
      // Personen-Erkennung durchführen
      const detections = model ? await detectObjects(frameFile, confidenceThreshold) : [];
      const personDetections = detections.filter(d => d.label.toLowerCase() === 'person');
      
      // Frame nur behalten wenn mindestens eine Person erkannt wurde
      if (personDetections.length > 0) {
        const url = URL.createObjectURL(blob);
        frames.push({
          file: frameFile,
          url,
          detections: personDetections,
          frameNumber,
          sourceVideo: videoFile.name,
        });
      } else {
        skippedFrames++;
        // Blob-URL nicht erstellen für übersprungene Frames (Speicher sparen)
      }
    }
    
    if (skippedFrames > 0) {
      console.log(`Chunk: ${skippedFrames} Frames ohne Person übersprungen`);
    }
    
    return frames;
  };

  // Extract frames from video with 5-minute chunking
  const extractFramesFromVideo = async (videoFile: File): Promise<ImageData[]> => {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      const videoUrl = URL.createObjectURL(videoFile);
      video.src = videoUrl;
      video.muted = true;
      video.playsInline = true;
      video.preload = 'metadata';
      
      console.log("Video wird geladen...", videoFile.name, videoFile.size);
      
      const processVideo = async () => {
        try {
          const duration = video.duration;
          console.log("Video Metadaten geladen:", { duration, width: video.videoWidth, height: video.videoHeight });
          
          if (!duration || !isFinite(duration) || duration <= 0) {
            toast.error("Video-Dauer konnte nicht ermittelt werden");
            resolve([]);
            return;
          }
          
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          
          if (!ctx) {
            toast.error("Canvas konnte nicht erstellt werden");
            resolve([]);
            return;
          }
          
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          
          const allFrames: ImageData[] = [];
          const totalChunks = Math.ceil(duration / CHUNK_DURATION);
          const totalFrames = Math.floor(duration / frameInterval);
          let processedFrames = 0;
          
          console.log(`Video: ${Math.floor(duration / 60)} Min, ${totalChunks} Chunks, ~${totalFrames} Frames`);
          toast.info(`Video: ${Math.floor(duration / 60)} Minuten → ${totalChunks} Chunk(s) à 5 Min`);
          
          for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
            const startTime = chunkIndex * CHUNK_DURATION;
            const endTime = Math.min((chunkIndex + 1) * CHUNK_DURATION, duration);
            const globalFrameOffset = Math.floor(startTime / frameInterval);
            
            console.log(`Chunk ${chunkIndex + 1}/${totalChunks}: ${startTime}s - ${endTime}s`);
            toast.loading(`Chunk ${chunkIndex + 1}/${totalChunks} wird verarbeitet (${Math.floor(startTime / 60)}-${Math.floor(endTime / 60)} Min)...`);
            
            const chunkFrames = await extractFramesFromChunk(
              video,
              canvas,
              ctx,
              videoFile,
              startTime,
              endTime,
              globalFrameOffset
            );
            
            allFrames.push(...chunkFrames);
            processedFrames += chunkFrames.length;
            
            setExtractionProgress(Math.round((processedFrames / (totalFrames + 1)) * 100));
            
            // Add extracted chunk frames to state immediately so user can see progress
            if (chunkIndex < totalChunks - 1) {
              setImages(prev => [...prev, ...chunkFrames]);
              toast.success(`Chunk ${chunkIndex + 1}/${totalChunks} fertig (${chunkFrames.length} Frames)`);
              
              // Small delay to let browser breathe
              await new Promise(r => setTimeout(r, 100));
            }
          }
          
          URL.revokeObjectURL(videoUrl);
          resolve(allFrames);
        } catch (error) {
          console.error("Fehler bei Video-Verarbeitung:", error);
          toast.error("Fehler bei der Video-Verarbeitung");
          resolve([]);
        }
      };
      
      video.onloadedmetadata = () => {
        console.log("onloadedmetadata triggered");
        // Wait for video to be actually ready to seek
        if (video.readyState >= 1) {
          processVideo();
        } else {
          video.oncanplay = () => {
            console.log("oncanplay triggered");
            processVideo();
          };
        }
      };
      
      video.onerror = (e) => {
        console.error("Video Ladefehler:", e);
        toast.error("Fehler beim Laden des Videos - Format wird möglicherweise nicht unterstützt");
        URL.revokeObjectURL(videoUrl);
        resolve([]);
      };
      
      // Force load
      video.load();
    });
  };

  const handleVideoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    if (!model) {
      toast.error("KI-Modell wird noch geladen, bitte warten...");
      return;
    }

    setIsProcessing(true);
    setExtractionProgress(0);
    toast.loading("Video wird in Frames zerlegt...");

    const allFrames: ImageData[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      toast.loading(`Video ${i + 1} von ${files.length} wird verarbeitet...`);
      const frames = await extractFramesFromVideo(file);
      allFrames.push(...frames);
    }

    setImages(allFrames);
    setCurrentIndex(0);
    setSelectedDetection(null);
    setIsProcessing(false);
    setExtractionProgress(0);
    
    const totalDetections = allFrames.reduce((sum, img) => sum + img.detections.length, 0);
    toast.success(`${allFrames.length} Frames mit Personen behalten, ${totalDetections} Personen erkannt! (Frames ohne Person wurden verworfen)`);
    
    // Reset input
    if (videoInputRef.current) {
      videoInputRef.current.value = '';
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    if (!model) {
      toast.error("KI-Modell wird noch geladen, bitte warten...");
      return;
    }

    setIsProcessing(true);
    toast.loading("Bilder werden automatisch analysiert...");

    const newImages: ImageData[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const url = URL.createObjectURL(file);
      
      const detections = await detectObjects(file, confidenceThreshold);
      
      newImages.push({
        file,
        url,
        detections,
      });
      
      toast.loading(`Verarbeite ${i + 1} von ${files.length}...`);
    }

    setImages(newImages);
    setCurrentIndex(0);
    setSelectedDetection(null);
    setIsProcessing(false);
    
    const totalDetections = newImages.reduce((sum, img) => sum + img.detections.length, 0);
    const uniqueLabels = [...new Set(newImages.flatMap(img => img.detections.map(d => d.label)))];
    toast.success(`${newImages.length} Bild(er) verarbeitet, ${totalDetections} Objekte erkannt! (${uniqueLabels.join(", ")})`);
  };

  // Re-detect with new threshold
  const handleRedetect = async () => {
    if (!model || images.length === 0) return;
    
    setIsProcessing(true);
    toast.loading("Erneute Analyse mit neuem Schwellenwert...");

    const newImages: ImageData[] = [];

    for (let i = 0; i < images.length; i++) {
      const img = images[i];
      const detections = await detectObjects(img.file, confidenceThreshold);
      
      newImages.push({
        ...img,
        detections,
      });
      
      toast.loading(`Verarbeite ${i + 1} von ${images.length}...`);
    }

    setImages(newImages);
    setSelectedDetection(null);
    setIsProcessing(false);
    
    const totalDetections = newImages.reduce((sum, img) => sum + img.detections.length, 0);
    toast.success(`${totalDetections} Objekte mit ${(confidenceThreshold * 100).toFixed(0)}% Schwellenwert erkannt!`);
  };

  const handleNext = () => {
    if (currentIndex < images.length - 1) {
      setCurrentIndex(currentIndex + 1);
      setSelectedDetection(null);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
      setSelectedDetection(null);
    }
  };

  const handleReset = () => {
    setImages([]);
    setCurrentIndex(0);
    setSelectedDetection(null);
    toast.success("Bilder zurückgesetzt!");
  };

  const handleExport = async () => {
    const zip = new JSZip();

    const imagesWithDetections = images.filter(img => img.detections.length > 0);
    
    if (imagesWithDetections.length === 0) {
      toast.error("Keine Bilder mit Erkennungen zum Exportieren!");
      return;
    }

    // Gruppiere Bilder nach Tätigkeitskategorie (Label der ersten Detection)
    const categoryFolders: { [category: string]: typeof imagesWithDetections } = {};
    
    for (const img of imagesWithDetections) {
      // Nimm das Label der ersten Detektion als Kategorie
      const category = img.detections[0]?.label || "Ohne_Kategorie";
      // Sanitize folder name (entferne Sonderzeichen, ersetze Leerzeichen)
      const safeCategoryName = category.replace(/[\/\\:*?"<>|]/g, "_").replace(/\s+/g, "_");
      
      if (!categoryFolders[safeCategoryName]) {
        categoryFolders[safeCategoryName] = [];
      }
      categoryFolders[safeCategoryName].push(img);
    }

    let exportedCount = 0;
    
    for (const [categoryName, categoryImages] of Object.entries(categoryFolders)) {
      // Erstelle Ordner für jede Kategorie
      const folder = zip.folder(categoryName);
      if (!folder) continue;
      
      for (let i = 0; i < categoryImages.length; i++) {
        const img = categoryImages[i];
        const imgElement = new Image();
        imgElement.src = img.url;

        await new Promise((resolve) => {
          imgElement.onload = () => {
            const labelBeeFormat = {
              width: imgElement.width,
              height: imgElement.height,
              valid: true,
              rotate: 0,
              step_1: {
                toolName: "rectTool",
                result: img.detections.map((detection, idx) => ({
                  x: detection.x,
                  y: detection.y,
                  width: detection.width,
                  height: detection.height,
                  attribute: detection.label,
                  valid: true,
                  id: `detect_${exportedCount}_${idx}`,
                  sourceID: "",
                  textAttribute: "",
                  order: idx + 1,
                })),
              },
            };

            const jsonFilename = img.file.name.replace(/\.[^/.]+$/, "") + ".json";
            folder.file(jsonFilename, JSON.stringify(labelBeeFormat, null, 2));
            exportedCount++;
            resolve(null);
          };
        });
      }
    }

    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "annotations.zip";
    a.click();
    
    const categoryCount = Object.keys(categoryFolders).length;
    const skipped = images.length - imagesWithDetections.length;
    const categoryList = Object.keys(categoryFolders).join(", ");
    
    if (skipped > 0) {
      toast.success(`${exportedCount} Annotationen in ${categoryCount} Ordner exportiert: ${categoryList} (${skipped} ohne Erkennung übersprungen)`);
    } else {
      toast.success(`${exportedCount} Annotationen in ${categoryCount} Ordner exportiert: ${categoryList}`);
    }
  };


  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground flex items-center gap-2">
              <Sparkles className="w-8 h-8 text-primary" />
              Auto-Annotator
            </h1>
            <p className="text-muted-foreground mt-1">
              Automatische Erkennung aller Objekte mit KI (COCO-SSD)
            </p>
          </div>
          
          <div className="flex gap-3 flex-wrap">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileUpload}
              className="hidden"
            />
            <input
              ref={videoInputRef}
              type="file"
              accept="video/*"
              multiple
              onChange={handleVideoUpload}
              className="hidden"
            />
            <Button
              onClick={() => fileInputRef.current?.click()}
              disabled={isProcessing}
              variant="default"
            >
              <Upload className="w-4 h-4 mr-2" />
              Bilder
            </Button>
            <Button
              onClick={() => videoInputRef.current?.click()}
              disabled={isProcessing}
              variant="secondary"
            >
              <Video className="w-4 h-4 mr-2" />
              Video
            </Button>
            
            {images.length > 0 && (
              <>
                <Button onClick={handleExport} variant="outline">
                  <Download className="w-4 h-4 mr-2" />
                  Exportieren
                </Button>
                <Button onClick={handleReset} variant="ghost">
                  Zurücksetzen
                </Button>
              </>
            )}
          </div>
        </div>

        {/* Progress bar for video extraction */}
        {isProcessing && extractionProgress > 0 && (
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <Film className="w-5 h-5 text-primary animate-pulse" />
              <div className="flex-1">
                <div className="flex justify-between text-sm mb-1">
                  <span>Frames werden extrahiert...</span>
                  <span>{extractionProgress}%</span>
                </div>
                <div className="w-full bg-secondary rounded-full h-2">
                  <div 
                    className="bg-primary h-2 rounded-full transition-all duration-300"
                    style={{ width: `${extractionProgress}%` }}
                  />
                </div>
              </div>
            </div>
          </Card>
        )}

        {/* Main Content */}
        {images.length === 0 ? (
          <Card className="p-12 text-center border-dashed border-2">
            <div className="flex justify-center gap-4 mb-6">
              <Upload className="w-12 h-12 text-muted-foreground" />
              <Video className="w-12 h-12 text-muted-foreground" />
            </div>
            <h3 className="text-xl font-semibold mb-2">Keine Medien geladen</h3>
            <p className="text-muted-foreground mb-6">
              Laden Sie Bilder oder Videos hoch, um die automatische Erkennung zu starten
            </p>
            <div className="flex gap-4 justify-center flex-wrap">
              <Button onClick={() => fileInputRef.current?.click()}>
                <Upload className="w-4 h-4 mr-2" />
                Bilder auswählen
              </Button>
              <Button onClick={() => videoInputRef.current?.click()} variant="secondary">
                <Video className="w-4 h-4 mr-2" />
                Video auswählen
              </Button>
            </div>
            <div className="mt-6 p-4 bg-secondary/50 rounded-lg max-w-lg mx-auto space-y-4">
              <div>
                <p className="text-sm text-muted-foreground mb-2">
                  <strong>Video-Einstellung:</strong> Frame-Intervall
                </p>
                <div className="flex items-center gap-3">
                  <span className="text-sm w-8">{frameInterval}s</span>
                  <Slider
                    value={[frameInterval]}
                    onValueChange={([v]) => setFrameInterval(v)}
                    min={0.5}
                    max={5}
                    step={0.5}
                    className="flex-1"
                  />
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Extrahiert einen Frame alle {frameInterval} Sekunden
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground mb-2">
                  <strong>Bounding Box Padding:</strong> {boundingBoxPadding}%
                </p>
                <div className="flex items-center gap-3">
                  <span className="text-sm w-8">{boundingBoxPadding}%</span>
                  <Slider
                    value={[boundingBoxPadding]}
                    onValueChange={([v]) => setBoundingBoxPadding(v)}
                    min={0}
                    max={50}
                    step={5}
                    className="flex-1"
                  />
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Erweitert Erkennungsboxen um {boundingBoxPadding}% links/rechts für Werkzeuge
                </p>
              </div>
            </div>
          </Card>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Canvas Area */}
            <Card className="lg:col-span-3 p-6 bg-canvas-bg">
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm text-muted-foreground">
                  Bild {currentIndex + 1} von {images.length}
                </span>
                <span className="text-sm text-primary font-medium">
                  {currentImage.detections.length} Erkennungen
                </span>
              </div>
              
              <div ref={containerRef} className="relative bg-black/20 rounded-lg overflow-hidden">
                <canvas
                  ref={canvasRef}
                  className="max-w-full h-auto mx-auto"
                  style={{ cursor: getCursorStyle() }}
                  onMouseDown={handleCanvasMouseDown}
                  onMouseMove={handleCanvasMouseMove}
                  onMouseUp={handleCanvasMouseUp}
                  onMouseLeave={handleCanvasMouseUp}
                />
              </div>

              {/* Navigation */}
              <div className="flex items-center justify-between mt-6">
                <Button
                  onClick={handlePrevious}
                  disabled={currentIndex === 0}
                  variant="secondary"
                >
                  <ChevronLeft className="w-4 h-4 mr-2" />
                  Zurück (A)
                </Button>
                
                <div className="text-center">
                  <span className="text-sm text-muted-foreground">
                    {currentImage.file.name}
                  </span>
                  {currentImage.sourceVideo && (
                    <span className="block text-xs text-primary mt-1">
                      <Film className="w-3 h-3 inline mr-1" />
                      Frame {(currentImage.frameNumber ?? 0) + 1} aus {currentImage.sourceVideo}
                    </span>
                  )}
                </div>
                
                <Button
                  onClick={handleNext}
                  disabled={currentIndex === images.length - 1}
                  variant="default"
                >
                  Weiter (D)
                  <ChevronRight className="w-4 h-4 ml-2" />
                </Button>
              </div>
            </Card>

            {/* Sidebar */}
            <Card className="p-6 space-y-4">
              {/* Auto Label Settings */}
              <div className="pb-4 border-b border-border">
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  <SlidersHorizontal className="w-4 h-4" />
                  Auto-Label Einstellungen
                </h3>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-muted-foreground">Konfidenz-Schwellenwert</span>
                      <span className="font-medium">{(confidenceThreshold * 100).toFixed(0)}%</span>
                    </div>
                    <Slider
                      value={[confidenceThreshold]}
                      onValueChange={([v]) => setConfidenceThreshold(v)}
                      min={0.1}
                      max={0.95}
                      step={0.05}
                      className="w-full"
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      Niedrigerer Wert = mehr Erkennungen
                    </p>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-muted-foreground">Video Frame-Intervall</span>
                      <span className="font-medium">{frameInterval}s</span>
                    </div>
                    <Slider
                      value={[frameInterval]}
                      onValueChange={([v]) => setFrameInterval(v)}
                      min={0.5}
                      max={5}
                      step={0.5}
                      className="w-full"
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      Frame alle {frameInterval} Sekunden extrahieren
                    </p>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-muted-foreground">Bounding Box Padding</span>
                      <span className="font-medium">{boundingBoxPadding}%</span>
                    </div>
                    <Slider
                      value={[boundingBoxPadding]}
                      onValueChange={([v]) => setBoundingBoxPadding(v)}
                      min={0}
                      max={50}
                      step={5}
                      className="w-full"
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      Erweitert Box um {boundingBoxPadding}% links/rechts
                    </p>
                  </div>
                  <Button
                    size="sm"
                    variant="secondary"
                    className="w-full"
                    onClick={handleRedetect}
                    disabled={isProcessing || images.length === 0}
                  >
                    <Sparkles className="w-3 h-3 mr-2" />
                    Erneut erkennen
                  </Button>
                </div>
              </div>


              {/* Batch Label Bereich */}
              <div className="pb-4 border-b border-border">
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  <Tag className="w-4 h-4" />
                  Bereich labeln
                </h3>
                <p className="text-xs text-muted-foreground mb-2">
                  Bild-Bereich auswählen und Label zuweisen
                </p>
                <div className="space-y-2">
                  <div className="flex gap-2 items-center">
                    <Input
                      type="number"
                      min={1}
                      max={images.length}
                      placeholder="Von"
                      className="text-sm w-20"
                      id="batch-from"
                    />
                    <span className="text-muted-foreground">-</span>
                    <Input
                      type="number"
                      min={1}
                      max={images.length}
                      placeholder="Bis"
                      className="text-sm w-20"
                      id="batch-to"
                    />
                  </div>
                  <select
                    id="batch-label-select"
                    className="w-full h-9 rounded-md border border-input bg-background px-3 py-1 text-sm"
                  >
                    <option value="">Label wählen...</option>
                    {activityLabels.map((label) => (
                      <option key={label} value={label}>{label}</option>
                    ))}
                  </select>
                  <Button
                    size="sm"
                    className="w-full"
                    disabled={isProcessing || !model}
                    onClick={async () => {
                      const fromInput = document.getElementById('batch-from') as HTMLInputElement;
                      const toInput = document.getElementById('batch-to') as HTMLInputElement;
                      const labelSelect = document.getElementById('batch-label-select') as HTMLSelectElement;
                      
                      const from = parseInt(fromInput.value) - 1; // Convert to 0-indexed
                      const to = parseInt(toInput.value) - 1;
                      const label = labelSelect.value;
                      
                      if (isNaN(from) || isNaN(to)) {
                        toast.error("Bitte Bereich eingeben (Von - Bis)");
                        return;
                      }
                      if (!label) {
                        toast.error("Bitte Label auswählen");
                        return;
                      }
                      if (from < 0 || to >= images.length || from > to) {
                        toast.error(`Ungültiger Bereich. Muss zwischen 1 und ${images.length} sein.`);
                        return;
                      }
                      
                      setIsProcessing(true);
                      const newImages = [...images];
                      let detectedCount = 0;
                      
                      for (let i = from; i <= to; i++) {
                        // Wenn keine Detektionen vorhanden, führe Personen-Erkennung durch
                        if (newImages[i].detections.length === 0 && model) {
                          const detections = await detectObjects(newImages[i].file, confidenceThreshold);
                          // Nur "Person" Detektionen behalten und mit dem Label versehen
                          const personDetections = detections
                            .filter(d => d.label.toLowerCase() === 'person')
                            .map(d => ({ ...d, label }));
                          newImages[i].detections = personDetections;
                          if (personDetections.length > 0) detectedCount++;
                        } else {
                          // Bestehende Detektionen nur umbenennen
                          newImages[i].detections = newImages[i].detections.map(d => ({
                            ...d,
                            label: label
                          }));
                        }
                      }
                      
                      setImages(newImages);
                      setIsProcessing(false);
                      toast.success(`Bilder ${from + 1}-${to + 1} auf "${label}" gesetzt (${to - from + 1} Bilder, ${detectedCount} neu erkannt)`);
                    }}
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="w-3 h-3 animate-spin mr-1" />
                        Erkenne Personen...
                      </>
                    ) : (
                      "Bereich labeln"
                    )}
                  </Button>
                </div>
                
                {/* Quick range buttons */}
                <div className="mt-3 pt-3 border-t border-border/50">
                  <p className="text-xs text-muted-foreground mb-2">Schnell-Bereiche:</p>
                  <div className="flex flex-wrap gap-1">
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-xs h-6 px-2"
                      onClick={() => {
                        const fromInput = document.getElementById('batch-from') as HTMLInputElement;
                        const toInput = document.getElementById('batch-to') as HTMLInputElement;
                        fromInput.value = "1";
                        toInput.value = String(Math.min(50, images.length));
                      }}
                    >
                      1-50
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-xs h-6 px-2"
                      onClick={() => {
                        const fromInput = document.getElementById('batch-from') as HTMLInputElement;
                        const toInput = document.getElementById('batch-to') as HTMLInputElement;
                        fromInput.value = "51";
                        toInput.value = String(Math.min(100, images.length));
                      }}
                    >
                      51-100
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-xs h-6 px-2"
                      onClick={() => {
                        const fromInput = document.getElementById('batch-from') as HTMLInputElement;
                        const toInput = document.getElementById('batch-to') as HTMLInputElement;
                        fromInput.value = String(currentIndex + 1);
                        toInput.value = String(images.length);
                      }}
                    >
                      Ab hier
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-xs h-6 px-2"
                      onClick={() => {
                        const fromInput = document.getElementById('batch-from') as HTMLInputElement;
                        const toInput = document.getElementById('batch-to') as HTMLInputElement;
                        fromInput.value = "1";
                        toInput.value = String(images.length);
                      }}
                    >
                      Alle
                    </Button>
                  </div>
                </div>
                
                {/* Bereich löschen */}
                <div className="mt-3 pt-3 border-t border-border/50">
                  <p className="text-xs text-muted-foreground mb-2">Bereich löschen:</p>
                  <div className="flex gap-2 items-center mb-2">
                    <Input
                      type="number"
                      min={1}
                      max={images.length}
                      placeholder="Von"
                      className="text-sm w-20"
                      id="delete-from"
                    />
                    <span className="text-muted-foreground">-</span>
                    <Input
                      type="number"
                      min={1}
                      max={images.length}
                      placeholder="Bis"
                      className="text-sm w-20"
                      id="delete-to"
                    />
                  </div>
                  <Button
                    size="sm"
                    variant="destructive"
                    className="w-full"
                    onClick={() => {
                      const fromInput = document.getElementById('delete-from') as HTMLInputElement;
                      const toInput = document.getElementById('delete-to') as HTMLInputElement;
                      
                      const from = parseInt(fromInput.value) - 1;
                      const to = parseInt(toInput.value) - 1;
                      
                      if (isNaN(from) || isNaN(to)) {
                        toast.error("Bitte Bereich eingeben (Von - Bis)");
                        return;
                      }
                      if (from < 0 || to >= images.length || from > to) {
                        toast.error(`Ungültiger Bereich. Muss zwischen 1 und ${images.length} sein.`);
                        return;
                      }
                      
                      const newImages = [...images];
                      const deleteCount = to - from + 1;
                      newImages.splice(from, deleteCount);
                      setImages(newImages);
                      
                      // Setze aktuellen Index zurück falls nötig
                      if (currentIndex >= newImages.length) {
                        setCurrentIndex(Math.max(0, newImages.length - 1));
                      }
                      setSelectedDetection(null);
                      
                      // Felder zurücksetzen
                      fromInput.value = '';
                      toInput.value = '';
                      
                      toast.success(`${deleteCount} Bilder gelöscht`);
                    }}
                  >
                    <Trash2 className="w-3 h-3 mr-1" />
                    Bereich löschen
                  </Button>
                </div>
              </div>

              {/* Quick Activity Labels */}
              <div className="pb-4 border-b border-border">
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  <Sparkles className="w-4 h-4" />
                  Schnell-Labels (Tätigkeiten)
                </h3>
                <div className="flex flex-wrap gap-1">
                  {activityLabels.map((label) => (
                    <Button
                      key={label}
                      size="sm"
                      variant="outline"
                      className="text-xs h-7 px-2"
                      disabled={selectedDetection === null}
                      onClick={() => {
                        if (selectedDetection !== null) {
                          const newImages = [...images];
                          newImages[currentIndex].detections[selectedDetection].label = label;
                          setImages(newImages);
                          toast.success(`Label auf "${label}" geändert`);
                        }
                      }}
                    >
                      {label}
                    </Button>
                  ))}
                </div>
                {selectedDetection === null && (
                  <p className="text-xs text-muted-foreground mt-2">
                    Wähle zuerst eine Box aus, um ein Schnell-Label zuzuweisen
                  </p>
                )}
                
                {/* Custom Label hinzufügen */}
                <div className="mt-3 pt-3 border-t border-border/50">
                  <p className="text-xs text-muted-foreground mb-2">Eigenes Label hinzufügen:</p>
                  <div className="flex gap-2">
                    <Input
                      value={newCustomLabel}
                      onChange={(e) => setNewCustomLabel(e.target.value)}
                      placeholder="Neues Label..."
                      className="text-sm h-7"
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && newCustomLabel.trim()) {
                          if (!activityLabels.includes(newCustomLabel.trim())) {
                            setCustomActivityLabels(prev => [...prev, newCustomLabel.trim()]);
                            toast.success(`Label "${newCustomLabel.trim()}" hinzugefügt`);
                          } else {
                            toast.error("Label existiert bereits");
                          }
                          setNewCustomLabel("");
                        }
                      }}
                    />
                    <Button
                      size="sm"
                      variant="secondary"
                      className="h-7 px-2"
                      onClick={() => {
                        if (newCustomLabel.trim()) {
                          if (!activityLabels.includes(newCustomLabel.trim())) {
                            setCustomActivityLabels(prev => [...prev, newCustomLabel.trim()]);
                            toast.success(`Label "${newCustomLabel.trim()}" hinzugefügt`);
                          } else {
                            toast.error("Label existiert bereits");
                          }
                          setNewCustomLabel("");
                        }
                      }}
                    >
                      +
                    </Button>
                  </div>
                  {customActivityLabels.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {customActivityLabels.map((label, idx) => (
                        <span 
                          key={idx} 
                          className="text-xs bg-primary/20 text-primary px-2 py-1 rounded flex items-center gap-1"
                        >
                          {label}
                          <button
                            onClick={() => {
                              setCustomActivityLabels(prev => prev.filter((_, i) => i !== idx));
                              toast.success(`Label "${label}" entfernt`);
                            }}
                            className="hover:text-destructive"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div>
                <h3 className="font-semibold mb-3">Erkannte Objekte</h3>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {currentImage.detections.map((detection, idx) => (
                    <div
                      key={idx}
                      onClick={() => {
                        setSelectedDetection(idx);
                        setEditingLabel(null);
                      }}
                      className={`p-3 rounded-lg text-sm cursor-pointer transition-colors ${
                        idx === selectedDetection 
                          ? "bg-primary/20 border border-primary" 
                          : "bg-secondary hover:bg-secondary/80"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        {editingLabel === idx ? (
                          <div className="flex items-center gap-1 flex-1">
                            <Input
                              value={editLabelValue}
                              onChange={(e) => setEditLabelValue(e.target.value)}
                              className="h-6 text-xs"
                              autoFocus
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                  const newImages = [...images];
                                  newImages[currentIndex].detections[idx].label = editLabelValue;
                                  setImages(newImages);
                                  setEditingLabel(null);
                                  toast.success(`Label geändert`);
                                } else if (e.key === 'Escape') {
                                  setEditingLabel(null);
                                }
                              }}
                              onClick={(e) => e.stopPropagation()}
                            />
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-6 w-6 p-0 text-green-600"
                              onClick={(e) => {
                                e.stopPropagation();
                                const newImages = [...images];
                                newImages[currentIndex].detections[idx].label = editLabelValue;
                                setImages(newImages);
                                setEditingLabel(null);
                                toast.success(`Label geändert`);
                              }}
                            >
                              <Check className="w-3 h-3" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-6 w-6 p-0"
                              onClick={(e) => {
                                e.stopPropagation();
                                setEditingLabel(null);
                              }}
                            >
                              <X className="w-3 h-3" />
                            </Button>
                          </div>
                        ) : (
                          <>
                            <div className="font-medium text-foreground flex-1">
                              {detection.label}
                            </div>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-6 w-6 p-0"
                              onClick={(e) => {
                                e.stopPropagation();
                                setEditingLabel(idx);
                                setEditLabelValue(detection.label);
                              }}
                            >
                              <Pencil className="w-3 h-3" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-6 w-6 p-0 text-destructive hover:text-destructive"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteDetection(idx);
                              }}
                            >
                              <Trash2 className="w-3 h-3" />
                            </Button>
                          </>
                        )}
                      </div>
                      <div className="text-muted-foreground text-xs mt-1">
                        Konfidenz: {(detection.confidence * 100).toFixed(0)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="pt-4 border-t border-border">
                <h3 className="font-semibold mb-2 text-sm">Tastenkürzel</h3>
                <div className="space-y-2 text-xs text-muted-foreground">
                  <div className="flex justify-between">
                    <span>Nächstes Bild:</span>
                    <kbd className="px-2 py-1 bg-muted rounded">D</kbd>
                  </div>
                  <div className="flex justify-between">
                    <span>Vorheriges Bild:</span>
                    <kbd className="px-2 py-1 bg-muted rounded">A</kbd>
                  </div>
                  <div className="flex justify-between">
                    <span>Löschen:</span>
                    <kbd className="px-2 py-1 bg-muted rounded">Entf</kbd>
                  </div>
                  <div className="flex justify-between">
                    <span>Abwählen:</span>
                    <kbd className="px-2 py-1 bg-muted rounded">Esc</kbd>
                  </div>
                </div>
              </div>
              
              {selectedDetection !== null && (
                <div className="pt-4 border-t border-border">
                  <p className="text-xs text-muted-foreground">
                    Klicke auf ✏️ um das Label zu ändern, oder wähle ein Schnell-Label oben.
                  </p>
                </div>
              )}
            </Card>
          </div>
        )}
      </div>
    </div>
  );
};
