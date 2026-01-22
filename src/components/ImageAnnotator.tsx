import { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Upload, ChevronLeft, ChevronRight, Download, Trash2, Tag, Sparkles, SlidersHorizontal, Video, Film, Pencil, Check, X, Loader2, FolderOpen, Scissors, RotateCcw } from "lucide-react";
import { toast } from "sonner";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";
import JSZip from "jszip";
import { ManualClipEditor } from "./ManualClipEditor";


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
  /** Frame-/Aktivitätslabel (wird beim Export priorisiert) */
  activityLabel?: string;
}

interface ClipData {
  name: string;
  frames: { file: File; url: string; frameNumber: number }[];
  sourceVideo: string;
  isManual?: boolean; // Flag for manually created clips - skip person detection
}

type ResizeHandle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w" | "move" | null;

export const ImageAnnotator = () => {
  const [activeTab, setActiveTab] = useState<"clips" | "annotator">("clips");
  
  // Clips erstellen state
  const [clips, setClips] = useState<ClipData[]>([]);
  const [clipFrameInterval, setClipFrameInterval] = useState(1);
  const [isCreatingClips, setIsCreatingClips] = useState(false);
  const [clipProgress, setClipProgress] = useState(0);
  const [framesPerClip, setFramesPerClip] = useState(32); // Default 32 frames per clip for YOWO
  
  // Annotator state
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
  
  // Manual box drawing state
  const [isDrawingBox, setIsDrawingBox] = useState(false);
  const [drawingStart, setDrawingStart] = useState<{ x: number; y: number } | null>(null);
  const [drawingPreview, setDrawingPreview] = useState<{ x: number; y: number; width: number; height: number } | null>(null);
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
  const [selectedImages, setSelectedImages] = useState<Set<number>>(new Set());
  const [filterMotionBlur, setFilterMotionBlur] = useState(true); // Filter out blurry/walking frames
  const [blurThreshold, setBlurThreshold] = useState(100); // Laplacian variance threshold
  const [chunkDuration, setChunkDuration] = useState(10); // Chunk duration in minutes (1-15)
  const [gpuBackend, setGpuBackend] = useState<string>("loading"); // Track which backend is active
  const [enhanceVideo, setEnhanceVideo] = useState(false); // Video upscaling/enhancement
  const [enhanceScale, setEnhanceScale] = useState(2); // Upscaling factor (1.5x - 4x)
  
  // Export settings
  const [exportOnlyLabeled, setExportOnlyLabeled] = useState(true); // Only export labeled frames
  
  // Persistenter Clip-Zähler pro Label (für fortlaufende Nummerierung über Videos hinweg)
  const [labelClipCounters, setLabelClipCounters] = useState<Record<string, number>>(() => {
    try {
      const saved = localStorage.getItem("labelClipCounters");
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  });
  
  // Speichere Clip-Zähler in localStorage
  useEffect(() => {
    localStorage.setItem("labelClipCounters", JSON.stringify(labelClipCounters));
  }, [labelClipCounters]);
  
  // Keyboard shortcuts for quick labeling (F, G, H keys)
  const [quickLabelF, setQuickLabelF] = useState("MAG-Schweißen");
  const [quickLabelG, setQuickLabelG] = useState("Putzen/Nacharbeiten");
  const [quickLabelH, setQuickLabelH] = useState("Winkelschleifer");
  
  
  // Preset activity labels for quick selection - Schweißumfeld Tätigkeiten
  const defaultActivityLabels = [
    "MAG-Schweißen", 
    "Putzen/Nacharbeiten", 
    "Winkelschleifer",
    "Transport", 
    "Zwischenkontrolle"
  ];
  
  // Kombiniere Standard-Labels mit benutzerdefinierten Labels
  const activityLabels = [...defaultActivityLabels, ...customActivityLabels];
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const clipVideoInputRef = useRef<HTMLInputElement>(null);
  const clipFolderInputRef = useRef<HTMLInputElement>(null);
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
          setGpuBackend("webgpu");
          console.log("✅ WebGPU backend activated (2-3x faster!)");
          toast.success("WebGPU aktiviert - GPU-Beschleunigung aktiv! 🚀");
        } catch (webgpuError) {
          console.warn("WebGPU not available, falling back to WebGL:", webgpuError);
          await tf.setBackend("webgl");
          await tf.ready();
          backend = "webgl";
          setGpuBackend("webgl");
          toast.info("WebGL aktiviert - WebGPU nicht verfügbar");
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
      // Ignore if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      
      // Only handle shortcuts in annotator tab
      if (activeTab !== "annotator") return;
      
      if (e.key === "d" || e.key === "D") {
        handleNext();
      } else if (e.key === "a" || e.key === "A") {
        handlePrevious();
      } else if (e.key === " ") {
        // Spacebar toggles selection for deletion
        e.preventDefault();
        if (images.length > 0) {
          setSelectedImages(prev => {
            const newSet = new Set(prev);
            if (newSet.has(currentIndex)) {
              newSet.delete(currentIndex);
            } else {
              newSet.add(currentIndex);
            }
            return newSet;
          });
        }
      } else if (e.key === "f" || e.key === "F") {
        // Quick label with F key
        applyQuickLabel(quickLabelF);
      } else if (e.key === "g" || e.key === "G") {
        // Quick label with G key
        applyQuickLabel(quickLabelG);
      } else if (e.key === "h" || e.key === "H") {
        // Quick label with H key
        applyQuickLabel(quickLabelH);
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
  }, [currentIndex, images.length, selectedDetection, quickLabelF, quickLabelG, quickLabelH, activeTab]);

  // Apply quick label to all detections on current image AND unmark from deletion
  const applyQuickLabel = (label: string) => {
    if (images.length === 0 || !currentImage) return;

    const newImages = [...images];
    newImages[currentIndex].activityLabel = label;
    newImages[currentIndex].detections = newImages[currentIndex].detections.map((det) => ({
      ...det,
      label,
    }));
    setImages(newImages);

    // Automatically unmark from deletion when labeled
    if (selectedImages.has(currentIndex)) {
      setSelectedImages((prev) => {
        const newSet = new Set(prev);
        newSet.delete(currentIndex);
        return newSet;
      });
    }

    toast.success(`Label "${label}" angewendet (demarkiert)`);
  };

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
      
      // Draw manual box preview while drawing
      if (drawingPreview) {
        ctx.strokeStyle = "hsl(120, 70%, 50%)"; // Green for new boxes
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 4]);
        ctx.strokeRect(drawingPreview.x, drawingPreview.y, drawingPreview.width, drawingPreview.height);
        ctx.setLineDash([]);
        
        // Draw "Neue Box" label
        ctx.fillStyle = "hsl(120, 70%, 50%)";
        ctx.fillRect(drawingPreview.x, drawingPreview.y - 25, 80, 25);
        ctx.fillStyle = "white";
        ctx.font = "14px system-ui, -apple-system, sans-serif";
        ctx.fillText("Neue Box", drawingPreview.x + 6, drawingPreview.y - 8);
      }
      
      // Update scale after drawing
      setTimeout(updateScale, 10);
    };
  }, [currentImage, selectedDetection, previewDetection, drawingPreview, updateScale]);

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
    
    // Clicked outside all detections - start drawing new box
    setSelectedDetection(null);
    setIsDrawingBox(true);
    setDrawingStart({ x, y });
    setDrawingPreview({ x, y, width: 0, height: 0 });
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / scale;
    const y = (e.clientY - rect.top) / scale;
    
    // Handle drawing new box
    if (isDrawingBox && drawingStart) {
      const newX = Math.min(drawingStart.x, x);
      const newY = Math.min(drawingStart.y, y);
      const width = Math.abs(x - drawingStart.x);
      const height = Math.abs(y - drawingStart.y);
      setDrawingPreview({ x: newX, y: newY, width, height });
      return;
    }
    
    // Handle resizing existing detection
    if (!resizing || !currentImage || selectedDetection === null) return;
    
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
    // Finish drawing new box
    if (isDrawingBox && drawingPreview && currentImage) {
      // Only create box if it's big enough (min 30x30)
      if (drawingPreview.width >= 30 && drawingPreview.height >= 30) {
        const newDetection: Detection = {
          x: drawingPreview.x,
          y: drawingPreview.y,
          width: drawingPreview.width,
          height: drawingPreview.height,
          label: "person", // Default label - user can change
          confidence: 1.0, // Manual = 100% confidence
        };
        
        const newImages = [...images];
        newImages[currentIndex].detections.push(newDetection);
        setImages(newImages);
        
        // Select the new detection
        setSelectedDetection(newImages[currentIndex].detections.length - 1);
        toast.success("Neue Box erstellt");
      }
      
      setIsDrawingBox(false);
      setDrawingStart(null);
      setDrawingPreview(null);
      return;
    }
    
    // Commit the preview to actual state (for resizing)
    if (resizing && previewDetection && selectedDetection !== null) {
      const newImages = [...images];
      newImages[currentIndex].detections[selectedDetection] = previewDetection;
      setImages(newImages);
    }
    setResizing(null);
    setPreviewDetection(null);
  };

  const getCursorStyle = (): string => {
    if (!currentImage || !canvasRef.current) return "crosshair"; // Default to crosshair for drawing
    
    if (isDrawingBox) return "crosshair";
    
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
          return "crosshair";
      }
    }
    
    return selectedDetection !== null ? "move" : "crosshair";
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

  // Video chunking - use state variable for duration
  const CHUNK_DURATION = chunkDuration * 60; // Convert minutes to seconds

  // Motion blur detection using Laplacian variance
  const detectMotionBlur = (canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D, detection: Detection): number => {
    // Extract the person region
    const { x, y, width, height } = detection;
    const safeX = Math.max(0, Math.floor(x));
    const safeY = Math.max(0, Math.floor(y));
    const safeWidth = Math.min(Math.floor(width), canvas.width - safeX);
    const safeHeight = Math.min(Math.floor(height), canvas.height - safeY);
    
    if (safeWidth <= 0 || safeHeight <= 0) return 999; // Invalid region, assume sharp
    
    const imageData = ctx.getImageData(safeX, safeY, safeWidth, safeHeight);
    const data = imageData.data;
    
    // Convert to grayscale and compute Laplacian variance
    const gray: number[] = [];
    for (let i = 0; i < data.length; i += 4) {
      gray.push(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    
    // Simplified Laplacian (3x3 kernel approximation)
    let sum = 0;
    let sumSq = 0;
    let count = 0;
    
    for (let row = 1; row < safeHeight - 1; row++) {
      for (let col = 1; col < safeWidth - 1; col++) {
        const idx = row * safeWidth + col;
        // Laplacian: center * 4 - neighbors
        const laplacian = 4 * gray[idx] - gray[idx - 1] - gray[idx + 1] - gray[idx - safeWidth] - gray[idx + safeWidth];
        sum += laplacian;
        sumSq += laplacian * laplacian;
        count++;
      }
    }
    
    if (count === 0) return 999;
    
    const mean = sum / count;
    const variance = (sumSq / count) - (mean * mean);
    
    return variance;
  };

  // Video/Image enhancement with upscaling and sharpening (local, no AI)
  const enhanceFrame = (
    sourceCanvas: HTMLCanvasElement,
    sourceCtx: CanvasRenderingContext2D,
    scaleFactor: number
  ): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } => {
    // Create upscaled canvas
    const enhancedCanvas = document.createElement('canvas');
    const enhancedCtx = enhancedCanvas.getContext('2d', { willReadFrequently: true });
    
    if (!enhancedCtx) {
      return { canvas: sourceCanvas, ctx: sourceCtx };
    }
    
    // Set new dimensions
    const newWidth = Math.round(sourceCanvas.width * scaleFactor);
    const newHeight = Math.round(sourceCanvas.height * scaleFactor);
    enhancedCanvas.width = newWidth;
    enhancedCanvas.height = newHeight;
    
    // Enable high-quality image smoothing
    enhancedCtx.imageSmoothingEnabled = true;
    enhancedCtx.imageSmoothingQuality = 'high';
    
    // Draw upscaled image
    enhancedCtx.drawImage(sourceCanvas, 0, 0, newWidth, newHeight);
    
    // Apply sharpening convolution filter
    const imageData = enhancedCtx.getImageData(0, 0, newWidth, newHeight);
    const data = imageData.data;
    const width = newWidth;
    const height = newHeight;
    
    // Sharpening kernel (unsharp mask approximation)
    // Center = 5, edges = -1, more aggressive sharpening
    const kernel = [
      0, -1, 0,
      -1, 5, -1,
      0, -1, 0
    ];
    
    // Create output buffer
    const output = new Uint8ClampedArray(data.length);
    
    // Apply convolution (skip edges)
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const pixelIndex = (y * width + x) * 4;
        
        for (let c = 0; c < 3; c++) { // RGB channels only
          let sum = 0;
          let ki = 0;
          
          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              const neighborIndex = ((y + ky) * width + (x + kx)) * 4 + c;
              sum += data[neighborIndex] * kernel[ki];
              ki++;
            }
          }
          
          output[pixelIndex + c] = Math.min(255, Math.max(0, sum));
        }
        output[pixelIndex + 3] = data[pixelIndex + 3]; // Alpha unchanged
      }
    }
    
    // Copy edges (unprocessed)
    for (let x = 0; x < width; x++) {
      const topIdx = x * 4;
      const bottomIdx = ((height - 1) * width + x) * 4;
      for (let c = 0; c < 4; c++) {
        output[topIdx + c] = data[topIdx + c];
        output[bottomIdx + c] = data[bottomIdx + c];
      }
    }
    for (let y = 0; y < height; y++) {
      const leftIdx = y * width * 4;
      const rightIdx = (y * width + width - 1) * 4;
      for (let c = 0; c < 4; c++) {
        output[leftIdx + c] = data[leftIdx + c];
        output[rightIdx + c] = data[rightIdx + c];
      }
    }
    
    // Apply enhanced data back
    const enhancedImageData = new ImageData(output, width, height);
    enhancedCtx.putImageData(enhancedImageData, 0, 0);
    
    // Additional contrast boost for better edge detection
    enhancedCtx.globalCompositeOperation = 'source-over';
    
    return { canvas: enhancedCanvas, ctx: enhancedCtx };
  };

  // ==================== CLIPS ERSTELLEN FUNKTIONEN ====================
  
  // Extract frames from video and create clips
  const createClipsFromVideo = async (videoFile: File): Promise<ClipData[]> => {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      const videoUrl = URL.createObjectURL(videoFile);
      video.src = videoUrl;
      video.muted = true;
      video.playsInline = true;
      video.preload = 'metadata';
      
      const processVideo = async () => {
        try {
          const duration = video.duration;
          
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
          
          // Calculate total frames
          const totalFrames = Math.floor(duration / clipFrameInterval);
          // Calculate number of clips needed
          const numClips = Math.ceil(totalFrames / framesPerClip);
          
          const createdClips: ClipData[] = [];
          let globalFrameNumber = 0;
          
          for (let clipIdx = 0; clipIdx < numClips; clipIdx++) {
            const clipFrames: { file: File; url: string; frameNumber: number }[] = [];
            const clipName = `${videoFile.name.replace(/\.[^/.]+$/, "")}${String(clipIdx + 1).padStart(2, '0')}`;
            
            for (let frameInClip = 0; frameInClip < framesPerClip; frameInClip++) {
              const globalFrameIdx = clipIdx * framesPerClip + frameInClip;
              const currentTime = globalFrameIdx * clipFrameInterval;
              
              if (currentTime > duration) break;
              
              video.currentTime = currentTime;
              
              await new Promise<void>((seekResolve) => {
                video.onseeked = () => seekResolve();
              });
              
              ctx.drawImage(video, 0, 0);
              
              const blob = await new Promise<Blob>((blobResolve) => {
                canvas.toBlob((b) => blobResolve(b!), 'image/jpeg', 0.95);
              });
              
              globalFrameNumber++;
              const frameName = `${String(frameInClip + 1).padStart(5, '0')}.jpg`;
              const frameFile = new File([blob], frameName, { type: 'image/jpeg' });
              
              clipFrames.push({
                file: frameFile,
                url: URL.createObjectURL(blob),
                frameNumber: frameInClip + 1
              });
              
              // Update progress
              setClipProgress(Math.round((globalFrameNumber / totalFrames) * 100));
            }
            
            if (clipFrames.length > 0) {
              createdClips.push({
                name: clipName,
                frames: clipFrames,
                sourceVideo: videoFile.name
              });
            }
          }
          
          URL.revokeObjectURL(videoUrl);
          resolve(createdClips);
        } catch (error) {
          console.error("Fehler bei Video-Verarbeitung:", error);
          toast.error("Fehler bei der Video-Verarbeitung");
          resolve([]);
        }
      };
      
      video.onloadedmetadata = () => {
        if (video.readyState >= 1) {
          processVideo();
        } else {
          video.oncanplay = () => processVideo();
        }
      };
      
      video.onerror = () => {
        toast.error("Fehler beim Laden des Videos");
        URL.revokeObjectURL(videoUrl);
        resolve([]);
      };
      
      video.load();
    });
  };

  const handleClipVideoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsCreatingClips(true);
    setClipProgress(0);
    toast.loading("Videos werden in Clips aufgeteilt...");

    const allClips: ClipData[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      toast.loading(`Video ${i + 1} von ${files.length}: ${file.name}`);
      const newClips = await createClipsFromVideo(file);
      allClips.push(...newClips);
    }

    setClips(prev => [...prev, ...allClips]);
    setIsCreatingClips(false);
    setClipProgress(0);
    
    const totalFrames = allClips.reduce((sum, c) => sum + c.frames.length, 0);
    toast.success(`${allClips.length} Clips mit insgesamt ${totalFrames} Frames erstellt!`);
    
    if (clipVideoInputRef.current) {
      clipVideoInputRef.current.value = '';
    }
  };

  const handleExportClips = async () => {
    if (clips.length === 0) {
      toast.error("Keine Clips zum Exportieren");
      return;
    }

    const zip = new JSZip();
    
    for (const clip of clips) {
      const clipFolder = zip.folder(clip.name);
      if (!clipFolder) continue;
      
      for (const frame of clip.frames) {
        const response = await fetch(frame.url);
        const blob = await response.blob();
        clipFolder.file(frame.file.name, blob);
      }
    }

    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "clips_export.zip";
    a.click();
    
    const totalFrames = clips.reduce((sum, c) => sum + c.frames.length, 0);
    toast.success(`${clips.length} Clips mit ${totalFrames} Frames exportiert!`);
  };

  const handleLoadClipsToAnnotator = async () => {
    if (clips.length === 0) {
      toast.error("Keine Clips vorhanden");
      return;
    }
    
    // Model is always required now since we run detection on all clips
    if (!model) {
      toast.error("KI-Modell wird noch geladen, bitte warten...");
      return;
    }
    
    const hasManualClips = clips.some(c => c.isManual);
    const hasAutoClips = clips.some(c => !c.isManual);
    
    setIsProcessing(true);
    setActiveTab("annotator");
    
    // Shorter toast messages
    toast.loading("Analysiere Clips...");
    
    const allImages: ImageData[] = [];
    const blurryIndices: number[] = [];
    let totalFrames = clips.reduce((sum, c) => sum + c.frames.length, 0);
    let processedFrames = 0;
    let manualFramesAdded = 0;
    let autoFramesAdded = 0;
    let skippedNoPersonFrames = 0;
    
    for (const clip of clips) {
      for (const frame of clip.frames) {
        // Run person detection for ALL clips (manual and auto)
        const detections = await detectObjects(frame.file, confidenceThreshold);
        const personDetections = detections.filter(d => d.label.toLowerCase() === 'person');
        
        // For MANUAL clips: KEEP ALL frames (even without persons), but still detect
        if (clip.isManual) {
          const currentFrameIndex = allImages.length;
          
          // Check for motion blur if enabled and persons detected
          if (filterMotionBlur && personDetections.length > 0) {
            const img = new Image();
            img.src = frame.url;
            await new Promise<void>(resolve => {
              img.onload = () => resolve();
              img.onerror = () => resolve();
            });
            
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            if (ctx && img.width > 0) {
              canvas.width = img.width;
              canvas.height = img.height;
              ctx.drawImage(img, 0, 0);
              
              let allBlurry = true;
              for (const detection of personDetections) {
                const blurScore = detectMotionBlur(canvas, ctx, detection);
                if (blurScore >= blurThreshold) {
                  allBlurry = false;
                  break;
                }
              }
              
              if (allBlurry) {
                blurryIndices.push(currentFrameIndex);
              }
            }
          }
          
          allImages.push({
            file: frame.file,
            url: frame.url,
            detections: personDetections, // Include detected bounding boxes!
            frameNumber: frame.frameNumber,
            sourceVideo: clip.name
          });
          manualFramesAdded++;
        } else {
          // For AUTO clips: run person detection like before
          const detections = await detectObjects(frame.file, confidenceThreshold);
          const personDetections = detections.filter(d => d.label.toLowerCase() === 'person');
          
          // Only keep frames with persons detected
          if (personDetections.length > 0) {
            const currentFrameIndex = allImages.length;
            
            // Check for motion blur if enabled
            if (filterMotionBlur) {
              const img = new Image();
              img.src = frame.url;
              await new Promise<void>(resolve => {
                img.onload = () => resolve();
                img.onerror = () => resolve();
              });
              
              const canvas = document.createElement('canvas');
              const ctx = canvas.getContext('2d');
              if (ctx && img.width > 0) {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                
                let allBlurry = true;
                for (const detection of personDetections) {
                  const blurScore = detectMotionBlur(canvas, ctx, detection);
                  if (blurScore >= blurThreshold) {
                    allBlurry = false;
                    break;
                  }
                }
                
                if (allBlurry) {
                  blurryIndices.push(currentFrameIndex);
                }
              }
            }
            
            allImages.push({
              file: frame.file,
              url: frame.url,
              detections: personDetections,
              frameNumber: frame.frameNumber,
              sourceVideo: clip.name
            });
            autoFramesAdded++;
          } else {
            skippedNoPersonFrames++;
          }
        }
        
        processedFrames++;
        if (processedFrames % 10 === 0) {
          setExtractionProgress(Math.round((processedFrames / totalFrames) * 100));
        }
      }
    }
    
    setImages(allImages);
    setSelectedImages(new Set(blurryIndices));
    setCurrentIndex(0);
    setSelectedDetection(null);
    setIsProcessing(false);
    setExtractionProgress(0);
    
    // Build detailed success message
    const parts: string[] = [];
    if (manualFramesAdded > 0) {
      parts.push(`${manualFramesAdded} manuelle Frames`);
    }
    if (autoFramesAdded > 0) {
      parts.push(`${autoFramesAdded} Auto-Frames mit Person`);
    }
    if (skippedNoPersonFrames > 0) {
      parts.push(`${skippedNoPersonFrames} ohne Person übersprungen`);
    }
    if (blurryIndices.length > 0) {
      parts.push(`${blurryIndices.length} unscharf markiert`);
    }
    
    toast.success(`${allImages.length} Frames geladen! (${parts.join(', ')})`);
  };

  // ==================== ANNOTATOR FUNKTIONEN ====================

  // Extract frames from a video chunk (startTime to endTime)
  const extractFramesFromChunk = async (
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D,
    videoFile: File,
    startTime: number,
    endTime: number,
    globalFrameOffset: number
  ): Promise<{ frames: ImageData[], blurryIndices: number[] }> => {
    const frames: ImageData[] = [];
    const blurryIndices: number[] = []; // Track indices of blurry frames
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
      
      // Apply video enhancement if enabled (upscale + sharpen for better detection)
      let detectionCanvas = canvas;
      let detectionCtx = ctx;
      
      if (enhanceVideo && enhanceScale > 1) {
        const enhanced = enhanceFrame(canvas, ctx, enhanceScale);
        detectionCanvas = enhanced.canvas;
        detectionCtx = enhanced.ctx;
      }
      
      // Convert enhanced or original canvas to blob for detection
      const detectionBlob = await new Promise<Blob>((blobResolve) => {
        detectionCanvas.toBlob((b) => blobResolve(b!), 'image/jpeg', 0.95);
      });
      
      const frameNumber = globalFrameOffset + i;
      const frameFile = new File([detectionBlob], `${videoFile.name}_frame_${frameNumber.toString().padStart(4, '0')}.jpg`, { type: 'image/jpeg' });
      
      // Personen-Erkennung auf dem (evtl. verbesserten) Frame durchführen
      const detections = model ? await detectObjects(frameFile, confidenceThreshold) : [];
      
      // Scale detections back to original coordinates if we enhanced the frame
      const scaledDetections = enhanceVideo && enhanceScale > 1
        ? detections.map(d => ({
            ...d,
            x: d.x / enhanceScale,
            y: d.y / enhanceScale,
            width: d.width / enhanceScale,
            height: d.height / enhanceScale,
          }))
        : detections;
      
      const personDetections = scaledDetections.filter(d => d.label.toLowerCase() === 'person');
      
      // Frame nur behalten wenn mindestens eine Person erkannt wurde
      if (personDetections.length > 0) {
        // Create blob from ORIGINAL canvas for storage (not upscaled)
        const storageBlob = await new Promise<Blob>((blobResolve) => {
          canvas.toBlob((b) => blobResolve(b!), 'image/jpeg', 0.95);
        });
        const url = URL.createObjectURL(storageBlob);
        const storageFile = new File([storageBlob], `${videoFile.name}_frame_${frameNumber.toString().padStart(4, '0')}.jpg`, { type: 'image/jpeg' });
        const currentFrameIndex = frames.length;
        
        // Motion blur check - mark blurry frames instead of skipping
        if (filterMotionBlur) {
          let allBlurry = true;
          for (const detection of personDetections) {
            const blurScore = detectMotionBlur(canvas, ctx, detection);
            if (blurScore >= blurThreshold) {
              allBlurry = false;
              break;
            }
          }
          
          if (allBlurry) {
            blurryIndices.push(currentFrameIndex);
            console.log(`Frame ${frameNumber}: Person(en) unscharf (in Bewegung), wird markiert`);
          }
        }
        
        frames.push({
          file: storageFile,
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
    
    return { frames, blurryIndices };
  };

  // Extract frames from video with 5-minute chunking
  const extractFramesFromVideo = async (videoFile: File): Promise<{ frames: ImageData[], blurryIndices: number[] }> => {
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
            resolve({ frames: [], blurryIndices: [] });
            return;
          }
          
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          
          if (!ctx) {
            toast.error("Canvas konnte nicht erstellt werden");
            resolve({ frames: [], blurryIndices: [] });
            return;
          }
          
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          
          const allFrames: ImageData[] = [];
          const allBlurryIndices: number[] = [];
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
            
            const chunkResult = await extractFramesFromChunk(
              video,
              canvas,
              ctx,
              videoFile,
              startTime,
              endTime,
              globalFrameOffset
            );
            
            // Adjust blurry indices to global index
            const baseIndex = allFrames.length;
            const adjustedBlurryIndices = chunkResult.blurryIndices.map(i => baseIndex + i);
            
            allFrames.push(...chunkResult.frames);
            allBlurryIndices.push(...adjustedBlurryIndices);
            processedFrames += chunkResult.frames.length;
            
            setExtractionProgress(Math.round((processedFrames / (totalFrames + 1)) * 100));
            
            // Add extracted chunk frames to state immediately so user can see progress
            if (chunkIndex < totalChunks - 1) {
              setImages(prev => [...prev, ...chunkResult.frames]);
              // Also mark blurry frames
              setSelectedImages(prev => {
                const newSet = new Set(prev);
                adjustedBlurryIndices.forEach(i => newSet.add(i));
                return newSet;
              });
              toast.success(`Chunk ${chunkIndex + 1}/${totalChunks} fertig (${chunkResult.frames.length} Frames)`);
              
              // Small delay to let browser breathe
              await new Promise(r => setTimeout(r, 100));
            }
          }
          
          URL.revokeObjectURL(videoUrl);
          resolve({ frames: allFrames, blurryIndices: allBlurryIndices });
        } catch (error) {
          console.error("Fehler bei Video-Verarbeitung:", error);
          toast.error("Fehler bei der Video-Verarbeitung");
          resolve({ frames: [], blurryIndices: [] });
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
        resolve({ frames: [], blurryIndices: [] });
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
    const allBlurryIndices: number[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      toast.loading(`Video ${i + 1} von ${files.length} wird verarbeitet...`);
      const result = await extractFramesFromVideo(file);
      
      // Adjust blurry indices to global index
      const baseIndex = allFrames.length;
      const adjustedBlurryIndices = result.blurryIndices.map(idx => baseIndex + idx);
      
      allFrames.push(...result.frames);
      allBlurryIndices.push(...adjustedBlurryIndices);
    }

    setImages(allFrames);
    // Automatically mark blurry frames
    setSelectedImages(new Set(allBlurryIndices));
    setCurrentIndex(0);
    setSelectedDetection(null);
    setIsProcessing(false);
    setExtractionProgress(0);
    
    const blurryCount = allBlurryIndices.length;
    const totalDetections = allFrames.reduce((sum, img) => sum + img.detections.length, 0);
    toast.success(`${allFrames.length} Frames, ${totalDetections} Personen erkannt! ${blurryCount > 0 ? `(${blurryCount} unscharfe Frames automatisch markiert)` : ''}`);
    
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
      const keepLabel = (img.activityLabel ?? "").trim();

      let detections = await detectObjects(img.file, confidenceThreshold);

      // Wichtig: wenn der Frame bereits gelabelt wurde, darf "Neu erkennen" das Label nicht wieder auf "Person" zurücksetzen.
      if (keepLabel) {
        detections = detections.map((d) =>
          d.label.toLowerCase() === "person" ? { ...d, label: keepLabel } : d
        );
      }

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
    // Vollständiges Zurücksetzen aller Zustände
    setImages([]);
    setCurrentIndex(0);
    setSelectedDetection(null);
    setSelectedImages(new Set());
    setResizing(null);
    setPreviewDetection(null);
    setEditingLabel(null);
    setEditLabelValue("");
    setExtractionProgress(0);
    setIsProcessing(false);
    toast.success("Bilder zurückgesetzt!");
  };
  
  const handleClipsReset = () => {
    // Vollständiges Zurücksetzen der Clips und aller zugehörigen Zustände
    setClips([]);
    setClipProgress(0);
    setIsCreatingClips(false);
    // Auch Annotator zurücksetzen um Überlappung zu vermeiden
    setImages([]);
    setCurrentIndex(0);
    setSelectedDetection(null);
    setSelectedImages(new Set());
    setResizing(null);
    setPreviewDetection(null);
    setEditingLabel(null);
    setEditLabelValue("");
    setExtractionProgress(0);
    setIsProcessing(false);
    toast.success("Alles zurückgesetzt!");
  };

  // Export (gewünschte Struktur): nach Tätigkeit (Label) und darunter nach Clip
  //
  // {LabelName}/
  //   {LabelName}01/
  //     frames00001.jpg
  //     frames00002.jpg
  //   {LabelName}02/
  //     ...
  // {LabelName} labeld/
  //   {LabelName}01/
  //     frames00001.json
  //     frames00002.json
  //   {LabelName}02/
  //     ...
  //
  // Hinweis: Clip-Ordner werden je Label als 01, 02, 03... durchnummeriert (stabile Reihenfolge nach Clip-Name).
  const handleExportAnnotatedClips = async () => {
    const zip = new JSZip();

    // Default labels that should NOT be exported (COCO-SSD labels)
    const cocoLabels = new Set(
      [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
      ].map((l) => l.toLowerCase())
    );

    // Windows-safe path segment
    const toSafePathSegment = (name: string) =>
      name
        .trim()
        .replace(/\\/g, "＼")
        .replace(/\//g, "／")
        .replace(/:/g, "：")
        .replace(/\*/g, "＊")
        .replace(/\?/g, "？")
        .replace(/"/g, "＂")
        .replace(/</g, "＜")
        .replace(/>/g, "＞")
        .replace(/\|/g, "｜")
        .replace(/\.mp4$/i, "")
        .replace(/\.avi$/i, "")
        .replace(/\.mov$/i, "")
        .replace(/\.mkv$/i, "")
        .replace(/\.webm$/i, "");

    const pad2 = (n: number) => String(n).padStart(2, "0");

    // Helper to get custom label from the frame (priority: explicit activityLabel, fallback: detections)
    const getCustomLabel = (img: ImageData): string | null => {
      const explicit = (img.activityLabel ?? "").trim();
      if (explicit) return explicit;

      for (const det of img.detections) {
        const label = (det.label ?? "").trim();
        if (label && !cocoLabels.has(label.toLowerCase())) {
          return label;
        }
      }
      return null;
    };

    // Filter images based on exportOnlyLabeled setting
    const imagesToExport = exportOnlyLabeled
      ? images
          .map((img, idx) => ({ img, idx }))
          .filter(({ img }) => getCustomLabel(img) !== null)
      : images.map((img, idx) => ({ img, idx }));

    if (imagesToExport.length === 0) {
      toast.error(
        exportOnlyLabeled
          ? "Keine gelabelten Frames zum Exportieren. Bitte erst labeln!"
          : "Keine Frames zum Exportieren."
      );
      return;
    }

    // Build label -> clip -> frames
    const byLabel = new Map<string, Map<string, ImageData[]>>();
    const unlabeledByClip = new Map<string, ImageData[]>();

    for (const { img } of imagesToExport) {
      const clipKey = (img.sourceVideo ?? "ungrouped").trim() || "ungrouped";
      const label = getCustomLabel(img);

      if (label) {
        if (!byLabel.has(label)) byLabel.set(label, new Map());
        const byClip = byLabel.get(label)!;
        if (!byClip.has(clipKey)) byClip.set(clipKey, []);
        byClip.get(clipKey)!.push(img);
      } else if (!exportOnlyLabeled) {
        if (!unlabeledByClip.has(clipKey)) unlabeledByClip.set(clipKey, []);
        unlabeledByClip.get(clipKey)!.push(img);
      }
    }

    const sortFrames = (frames: ImageData[]) =>
      frames.sort((a, b) => {
        const fa = a.frameNumber ?? 0;
        const fb = b.frameNumber ?? 0;
        if (fa !== fb) return fa - fb;
        return a.file.name.localeCompare(b.file.name);
      });

    // Stable ordering inside each clip
    for (const [, byClip] of byLabel) {
      for (const [, frames] of byClip) sortFrames(frames);
    }
    for (const [, frames] of unlabeledByClip) sortFrames(frames);

    let totalExportedFrames = 0;
    let totalExportedClips = 0;

    // Track new clip counters for this export
    const newClipCounters: Record<string, number> = { ...labelClipCounters };

    for (const [label, byClip] of byLabel) {
      const safeLabel = toSafePathSegment(label);

      const jpgRoot = zip.folder(safeLabel);
      const jsonRoot = zip.folder(`${safeLabel} labeld`);
      if (!jpgRoot || !jsonRoot) continue;

      // Hole den aktuellen Clip-Zähler für dieses Label (oder starte bei 0)
      let currentClipOffset = newClipCounters[label] || 0;

      const clipKeys = Array.from(byClip.keys()).sort((a, b) => a.localeCompare(b));

      const MAX_FRAMES_PER_CLIP = 200;
      let totalClipsForLabel = 0;

      for (let clipIndex = 0; clipIndex < clipKeys.length; clipIndex++) {
        const clipKey = clipKeys[clipIndex];
        const frames = byClip.get(clipKey) ?? [];
        if (frames.length === 0) continue;

        // Teile Frames in Chunks von max 200 auf
        const frameChunks: ImageData[][] = [];
        for (let i = 0; i < frames.length; i += MAX_FRAMES_PER_CLIP) {
          frameChunks.push(frames.slice(i, i + MAX_FRAMES_PER_CLIP));
        }

        for (let chunkIndex = 0; chunkIndex < frameChunks.length; chunkIndex++) {
          const chunkFrames = frameChunks[chunkIndex];
          totalExportedClips++;
          totalClipsForLabel++;

          // Fortlaufende Clip-Nummerierung über Videos hinweg
          const clipNumber = currentClipOffset + totalClipsForLabel;
          const clipFolderName = `${safeLabel}${pad2(clipNumber)}`;
          const jpgClipFolder = jpgRoot.folder(toSafePathSegment(clipFolderName));
          const jsonClipFolder = jsonRoot.folder(toSafePathSegment(clipFolderName));
          if (!jpgClipFolder || !jsonClipFolder) continue;

          // Frame-Zähler beginnt bei JEDEM CLIP neu bei 1
          let clipFrameCounter = 0;

          for (const img of chunkFrames) {
            clipFrameCounter++;
            const baseName = `frames${String(clipFrameCounter).padStart(5, "0")}`;

            // JPG
            try {
              const response = await fetch(img.url);
              const blob = await response.blob();
              jpgClipFolder.file(`${baseName}.jpg`, blob);
            } catch (e) {
              console.warn("Failed to export frame image", img.file?.name, e);
            }

            // JSON
            const customDetections = img.detections.filter((d) => {
              const l = (d.label ?? "").trim();
              if (!l) return false;
              return !cocoLabels.has(l.toLowerCase());
            });

            const dims = await new Promise<{ width: number | null; height: number | null }>((resolve) => {
              const imgElement = new Image();
              imgElement.src = img.url;
              imgElement.onload = () => resolve({ width: imgElement.width, height: imgElement.height });
              imgElement.onerror = () => resolve({ width: null, height: null });
            });

            const frameData = {
              width: dims.width,
              height: dims.height,
              sourceClip: clipKey,
              clipFolder: clipFolderName,
              frameNumber: img.frameNumber ?? null,
              activityLabel: label,
              annotations: customDetections.map((det) => ({
                label: det.label,
                x: Math.round(det.x),
                y: Math.round(det.y),
                width: Math.round(det.width),
                height: Math.round(det.height),
                confidence: det.confidence,
              })),
            };

            jsonClipFolder.file(`${baseName}.json`, JSON.stringify(frameData, null, 2));

            totalExportedFrames++;
          }
        }
      }
      
      // Update den Clip-Zähler für dieses Label
      newClipCounters[label] = currentClipOffset + totalClipsForLabel;
    }
    
    // Speichere die neuen Clip-Zähler
    setLabelClipCounters(newClipCounters);

    // Optional: export unlabeled frames if exportOnlyLabeled is off
    if (!exportOnlyLabeled && unlabeledByClip.size > 0) {
      const unlabeledRoot = zip.folder("Unlabeled");
      if (unlabeledRoot) {
        const clipKeys = Array.from(unlabeledByClip.keys()).sort((a, b) => a.localeCompare(b));
        for (const clipKey of clipKeys) {
          const frames = unlabeledByClip.get(clipKey) ?? [];
          if (frames.length === 0) continue;

          const clipFolder = unlabeledRoot.folder(toSafePathSegment(clipKey));
          if (!clipFolder) continue;

          let frameCounter = 0;
          for (const img of frames) {
            frameCounter++;
            const baseName = String(frameCounter).padStart(5, "0");
            try {
              const response = await fetch(img.url);
              const blob = await response.blob();
              clipFolder.file(`${baseName}.jpg`, blob);
              totalExportedFrames++;
            } catch (e) {
              console.warn("Failed to export unlabeled frame", img.file?.name, e);
            }
          }
        }
      }
    }

    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "clips_export.zip";
    a.click();

    toast.success(
      `Export: ${totalExportedClips} Clips, ${totalExportedFrames} Frames${exportOnlyLabeled ? " (nur gelabelt)" : ""}`
    );
  };

  // Export structure (exact as requested):
  // {LabelName}/
  //   {LabelName}_labeled/{LabelName}01.json
  //   {LabelName}_unlabeled/{LabelName}01.jpg
  // (paired by identical base-name; marked images are excluded)
  const handleExport = async () => {
    const zip = new JSZip();

    // Default labels that should NOT be exported (COCO-SSD labels)
    const cocoLabels = new Set(
      [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
      ].map((l) => l.toLowerCase())
    );

    // Windows-safe path segment but keeps the visual look (e.g. "/" -> "／")
    const toSafePathSegment = (name: string) =>
      name
        .trim()
        .replace(/\\/g, "＼")
        .replace(/\//g, "／")
        .replace(/:/g, "：")
        .replace(/\*/g, "＊")
        .replace(/\?/g, "？")
        .replace(/"/g, "＂")
        .replace(/</g, "＜")
        .replace(/>/g, "＞")
        .replace(/\|/g, "｜");

    // Get all non-deleted images (exclude marked ones)
    const validImages = images
      .map((img, idx) => ({ img, idx }))
      .filter(({ idx }) => !selectedImages.has(idx));

    if (validImages.length === 0) {
      toast.error("Keine Frames zum Exportieren (alle markiert oder keine vorhanden)");
      return;
    }

    // Determine activity label from detections (first non-COCO label)
    const getCustomLabel = (img: ImageData): string | null => {
      for (const det of img.detections) {
        const raw = (det.label ?? "").trim();
        if (!raw) continue;
        if (!cocoLabels.has(raw.toLowerCase())) return raw;
      }
      return null;
    };

    const labeledItems = validImages
      .map(({ img, idx }) => ({ img, idx, label: getCustomLabel(img) }))
      .filter((x): x is { img: ImageData; idx: number; label: string } => x.label !== null);

    if (labeledItems.length === 0) {
      toast.error("Keine gelabelten Frames vorhanden! Bitte zuerst Labels vergeben (F/G/H Tasten).");
      return;
    }

    // Group by label
    const byLabel = new Map<string, { img: ImageData; idx: number }[]>();
    for (const item of labeledItems) {
      if (!byLabel.has(item.label)) byLabel.set(item.label, []);
      byLabel.get(item.label)!.push({ img: item.img, idx: item.idx });
    }

    // Stable ordering: clip name then frame number
    byLabel.forEach((items) => {
      items.sort((a, b) => {
        const clipA = a.img.sourceVideo ?? "";
        const clipB = b.img.sourceVideo ?? "";
        if (clipA !== clipB) return clipA.localeCompare(clipB);
        return (a.img.frameNumber ?? 0) - (b.img.frameNumber ?? 0);
      });
    });

    let totalExported = 0;

    for (const [label, items] of byLabel) {
      const safeLabel = toSafePathSegment(label);

      // New structure:
      // Label/
      //   Label01.jpg
      //   Label02.jpg
      // Label labeled/
      //   Label01.json
      //   Label02.json
      const jpgFolder = zip.folder(safeLabel);
      const jsonFolder = zip.folder(`${safeLabel} labeled`);
      if (!jpgFolder || !jsonFolder) continue;

      let counter = 0;

      for (const { img } of items) {
        counter++;
        totalExported++;

        const baseName = `${safeLabel}${String(counter).padStart(2, "0")}`;

        // Write JPG to Label folder
        const response = await fetch(img.url);
        const blob = await response.blob();
        jpgFolder.file(`${baseName}.jpg`, blob);

        // Write JSON to Label labeled folder
        const customDetections = img.detections.filter((d) => {
          const l = (d.label ?? "").trim();
          if (!l) return false;
          return !cocoLabels.has(l.toLowerCase());
        });

        await new Promise<void>((resolve) => {
          const imgElement = new Image();
          imgElement.src = img.url;
          imgElement.onload = () => {
            const annotation = {
              image: `${baseName}.jpg`,
              width: imgElement.width,
              height: imgElement.height,
              sourceVideo: img.sourceVideo ?? null,
              frameNumber: img.frameNumber ?? null,
              annotations: customDetections.map((det) => ({
                label: det.label,
                x: Math.round(det.x),
                y: Math.round(det.y),
                width: Math.round(det.width),
                height: Math.round(det.height),
                confidence: det.confidence,
              })),
            };
            jsonFolder.file(`${baseName}.json`, JSON.stringify(annotation, null, 2));
            resolve();
          };
          imgElement.onerror = () => resolve();
        });
      }
    }

    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "dataset_export.zip";
    a.click();

    const skipped = selectedImages.size;
    const labels = Array.from(byLabel.keys());
    toast.success(`Export: ${labels.join(", ")} (${totalExported} Dateien, ${skipped} übersprungen)`);
  };

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground flex items-center gap-2">
              <Sparkles className="w-8 h-8 text-primary" />
              YOWO Datensatz-Tool
            </h1>
            <p className="text-muted-foreground mt-1">
              Clips erstellen & Labels vergeben für Aktivitätserkennung
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as "clips" | "annotator")}>
          <TabsList className="grid w-full grid-cols-2 max-w-md">
            <TabsTrigger value="clips" className="flex items-center gap-2">
              <Scissors className="w-4 h-4" />
              Clips erstellen
            </TabsTrigger>
            <TabsTrigger value="annotator" className="flex items-center gap-2">
              <Tag className="w-4 h-4" />
              Auto-Annotator
            </TabsTrigger>
          </TabsList>

          {/* ==================== CLIPS ERSTELLEN TAB ==================== */}
          <TabsContent value="clips" className="space-y-6">
            <Card className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold flex items-center gap-2">
                  <FolderOpen className="w-5 h-5" />
                  Video zu Clips
                </h2>
                <div className="flex gap-3">
                  <input
                    ref={clipVideoInputRef}
                    type="file"
                    accept="video/*"
                    multiple
                    onChange={handleClipVideoUpload}
                    className="hidden"
                  />
                  <Button
                    onClick={() => clipVideoInputRef.current?.click()}
                    disabled={isCreatingClips}
                  >
                    <Video className="w-4 h-4 mr-2" />
                    Video(s) auswählen
                  </Button>
                  {clips.length > 0 && (
                    <>
                      <Button onClick={handleLoadClipsToAnnotator} variant="default" disabled={isProcessing || !model}>
                        {isProcessing ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            Analysiere...
                          </>
                        ) : (
                          <>
                            <Sparkles className="w-4 h-4 mr-2" />
                            Weiter zum Annotator
                          </>
                        )}
                      </Button>
                      <Button onClick={handleClipsReset} variant="ghost">
                        Zurücksetzen
                      </Button>
                    </>
                  )}
                </div>
              </div>

              {/* Settings */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-muted-foreground font-medium">Frame-Intervall</span>
                      <span className="font-bold">{clipFrameInterval}s</span>
                    </div>
                    <Slider
                      value={[clipFrameInterval]}
                      onValueChange={([v]) => setClipFrameInterval(v)}
                      min={0.5}
                      max={5}
                      step={0.5}
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      Extrahiert einen Frame alle {clipFrameInterval} Sekunden
                    </p>
                  </div>
                </div>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-muted-foreground font-medium">Frames pro Clip</span>
                      <span className="font-bold">{framesPerClip}</span>
                    </div>
                    <Slider
                      value={[framesPerClip]}
                      onValueChange={([v]) => setFramesPerClip(v)}
                      min={8}
                      max={64}
                      step={8}
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      YOWO Standard: 32 Frames pro Clip
                    </p>
                  </div>
                  
                  {/* Export Settings */}
                  <div className="mt-4 pt-4 border-t border-border/50">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium">Export: Nur gelabelte Frames</p>
                        <p className="text-xs text-muted-foreground">
                          Ungelabelte Frames werden ignoriert
                        </p>
                      </div>
                      <Button
                        variant={exportOnlyLabeled ? "default" : "outline"}
                        size="sm"
                        onClick={() => setExportOnlyLabeled(!exportOnlyLabeled)}
                      >
                        {exportOnlyLabeled ? "Aktiv" : "Aus"}
                      </Button>
                    </div>
                    <p className="text-xs text-muted-foreground mt-2 p-2 bg-secondary/50 rounded">
                      Struktur: <code>ClipXX/LabelName/Labeled/*.json</code> + <code>Unlabeled/*.jpg</code>
                    </p>
                  </div>
                </div>
              </div>

              {/* Progress */}
              {isCreatingClips && clipProgress > 0 && (
                <div className="mb-6">
                  <div className="flex items-center gap-3">
                    <Loader2 className="w-5 h-5 animate-spin text-primary" />
                    <div className="flex-1">
                      <div className="flex justify-between text-sm mb-1">
                        <span>Clips werden erstellt...</span>
                        <span>{clipProgress}%</span>
                      </div>
                      <div className="w-full bg-secondary rounded-full h-2">
                        <div 
                          className="bg-primary h-2 rounded-full transition-all duration-300"
                          style={{ width: `${clipProgress}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Clips List */}
              {clips.length === 0 ? (
                <div className="text-center py-12 border-2 border-dashed rounded-lg">
                  <Video className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">Keine Clips</h3>
                  <p className="text-muted-foreground mb-4">
                    Laden Sie ein Video hoch, um es automatisch in Clips aufzuteilen
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Jeder Clip enthält {framesPerClip} Frames im Abstand von {clipFrameInterval}s
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-sm text-muted-foreground">
                    <span>{clips.length} Clips erstellt</span>
                    <span>{clips.reduce((sum, c) => sum + c.frames.length, 0)} Frames gesamt</span>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 max-h-96 overflow-y-auto">
                    {clips.map((clip, idx) => (
                      <div key={idx} className={`rounded-lg p-3 text-center ${clip.isManual ? 'bg-primary/20 border border-primary/40' : 'bg-secondary'}`}>
                        <div className="text-xs text-muted-foreground mb-1">
                          Clip {idx + 1} {clip.isManual && <span className="text-primary">(manuell)</span>}
                        </div>
                        <div className="font-medium text-sm truncate" title={clip.name}>{clip.name}</div>
                        <div className="text-xs text-primary mt-1">{clip.frames.length} Frames</div>
                        {clip.frames[0] && (
                          <img 
                            src={clip.frames[0].url} 
                            alt={clip.name}
                            className="w-full h-16 object-cover rounded mt-2"
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </Card>

            {/* Manual Clip Editor */}
            <ManualClipEditor
              framesPerClip={framesPerClip}
              frameInterval={clipFrameInterval}
              confidenceThreshold={confidenceThreshold}
              setConfidenceThreshold={setConfidenceThreshold}
              boundingBoxPadding={boundingBoxPadding}
              setBoundingBoxPadding={setBoundingBoxPadding}
              filterMotionBlur={filterMotionBlur}
              setFilterMotionBlur={setFilterMotionBlur}
              blurThreshold={blurThreshold}
              setBlurThreshold={setBlurThreshold}
              enhanceVideo={enhanceVideo}
              setEnhanceVideo={setEnhanceVideo}
              enhanceScale={enhanceScale}
              setEnhanceScale={setEnhanceScale}
              onClipsCreated={(newClips) => {
                // Mark manual clips with isManual flag - but still run person detection!
                const markedClips = newClips.map(clip => ({ ...clip, isManual: true }));
                setClips(prev => [...prev, ...markedClips]);
                toast.success(`${newClips.length} manuelle Clips erstellt`);
              }}
            />
          </TabsContent>

          {/* ==================== ANNOTATOR TAB ==================== */}
          <TabsContent value="annotator" className="space-y-6">
            {/* Action Buttons */}
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
                  <Button onClick={handleExportAnnotatedClips} variant="default">
                    <Download className="w-4 h-4 mr-2" />
                    Export
                  </Button>
                  <Button onClick={handleReset} variant="ghost">
                    Zurücksetzen
                  </Button>
                </>
              )}
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
                  Laden Sie Bilder oder Videos hoch, oder nutzen Sie "Clips erstellen" zuerst
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
                  <Button onClick={() => setActiveTab("clips")} variant="outline">
                    <Scissors className="w-4 h-4 mr-2" />
                    Clips erstellen
                  </Button>
                </div>
                <div className="mt-6 p-4 bg-secondary/50 rounded-lg max-w-lg mx-auto space-y-4">
                  {/* GPU Status */}
                  <div className="flex items-center justify-between p-2 rounded bg-secondary/80">
                    <span className="text-sm font-medium">GPU-Backend:</span>
                    <span className={`text-sm font-bold ${gpuBackend === 'webgpu' ? 'text-green-500' : gpuBackend === 'webgl' ? 'text-yellow-500' : 'text-muted-foreground'}`}>
                      {gpuBackend === 'loading' ? '⏳ Lädt...' : gpuBackend === 'webgpu' ? '✅ WebGPU (GPU)' : '⚠️ WebGL (Fallback)'}
                    </span>
                  </div>
                  
                  {/* Konfidenz-Schwellenwert - moved to front */}
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-muted-foreground font-medium">Konfidenz-Schwellenwert</span>
                      <span className="font-bold">{(confidenceThreshold * 100).toFixed(0)}%</span>
                    </div>
                    <Slider
                      value={[confidenceThreshold]}
                      onValueChange={([v]) => setConfidenceThreshold(v)}
                      min={0.1}
                      max={0.95}
                      step={0.05}
                      className="flex-1"
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      Niedrigerer Wert = mehr Erkennungen, höher = genauer
                    </p>
                  </div>

                  {/* Video Chunk Duration */}
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-muted-foreground font-medium">Video-Chunk Größe</span>
                      <span className="font-bold">{chunkDuration} Min</span>
                    </div>
                    <Slider
                      value={[chunkDuration]}
                      onValueChange={([v]) => setChunkDuration(v)}
                      min={1}
                      max={15}
                      step={1}
                      className="flex-1"
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      Größere Chunks = schneller, aber mehr RAM-Verbrauch
                    </p>
                  </div>

                  <div>
                    <p className="text-sm text-muted-foreground mb-2">
                      <strong>Frame-Intervall</strong>
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
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-sm text-muted-foreground">
                        <strong>Bewegungsunschärfe filtern:</strong>
                      </p>
                      <Button
                        variant={filterMotionBlur ? "default" : "outline"}
                        size="sm"
                        onClick={() => setFilterMotionBlur(!filterMotionBlur)}
                      >
                        {filterMotionBlur ? "Aktiv" : "Aus"}
                      </Button>
                    </div>
                    {filterMotionBlur && (
                      <>
                        <div className="flex items-center gap-3">
                          <span className="text-sm w-10">{blurThreshold}</span>
                          <Slider
                            value={[blurThreshold]}
                            onValueChange={([v]) => setBlurThreshold(v)}
                            min={50}
                            max={300}
                            step={10}
                            className="flex-1"
                          />
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          Höher = weniger streng (mehr Bilder behalten). Niedrig = strenger (mehr unscharfe Bilder filtern)
                        </p>
                      </>
                    )}
                  </div>
                  
                  {/* Video Enhancement / Upscaling */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div>
                        <p className="text-sm font-medium">
                          Video-Verbesserung (Upscaling)
                        </p>
                        <p className="text-xs text-muted-foreground">
                          Verbessert schlechte Videoqualität
                        </p>
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
                      <>
                        <div className="flex items-center gap-3">
                          <span className="text-sm w-10">{enhanceScale}x</span>
                          <Slider
                            value={[enhanceScale]}
                            onValueChange={([v]) => setEnhanceScale(v)}
                            min={1.5}
                            max={4}
                            step={0.5}
                            className="flex-1"
                          />
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          Höher = bessere Details für KI, aber langsamer. 2x empfohlen.
                        </p>
                      </>
                    )}
                  </div>
                  
                  {/* Export Settings */}
                  <div className="pt-4 border-t border-border/50">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium">Export: Nur gelabelte Frames</p>
                        <p className="text-xs text-muted-foreground">
                          Ungelabelte werden ignoriert
                        </p>
                      </div>
                      <Button
                        variant={exportOnlyLabeled ? "default" : "outline"}
                        size="sm"
                        onClick={() => setExportOnlyLabeled(!exportOnlyLabeled)}
                      >
                        {exportOnlyLabeled ? "Aktiv" : "Aus"}
                      </Button>
                    </div>
                    <p className="text-xs text-muted-foreground mt-2 p-2 bg-secondary/50 rounded">
                      Struktur: <code className="text-primary">ClipXX/LabelName/Labeled/*.json</code> + <code className="text-primary">Unlabeled/*.jpg</code>
                    </p>
                  </div>
                </div>
              </Card>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Canvas Area */}
                <Card className="lg:col-span-3 p-6 bg-canvas-bg">
                  {/* Quick Labels with Keyboard Shortcuts */}
                  <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-muted-foreground">
                        Bild {currentIndex + 1} von {images.length}
                      </span>
                      {selectedImages.has(currentIndex) && (
                        <span className="text-xs bg-destructive/20 text-destructive px-2 py-0.5 rounded-full font-medium">
                          ✓ Markiert
                        </span>
                      )}
                    </div>
                    
                    {/* Quick Label Buttons */}
                    <div className="flex items-center gap-2">
                      <div className="flex items-center gap-1">
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 text-xs px-2 font-bold"
                          onClick={() => applyQuickLabel(quickLabelF)}
                        >
                          <span className="bg-primary/20 text-primary px-1 rounded mr-1">F</span>
                          {quickLabelF}
                        </Button>
                        <select
                          value={quickLabelF}
                          onChange={(e) => setQuickLabelF(e.target.value)}
                          className="h-7 w-6 rounded border border-input bg-background text-xs cursor-pointer"
                          title="F-Taste ändern"
                        >
                          {activityLabels.map(label => (
                            <option key={label} value={label}>{label}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div className="flex items-center gap-1">
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 text-xs px-2 font-bold"
                          onClick={() => applyQuickLabel(quickLabelG)}
                        >
                          <span className="bg-primary/20 text-primary px-1 rounded mr-1">G</span>
                          {quickLabelG}
                        </Button>
                        <select
                          value={quickLabelG}
                          onChange={(e) => setQuickLabelG(e.target.value)}
                          className="h-7 w-6 rounded border border-input bg-background text-xs cursor-pointer"
                          title="G-Taste ändern"
                        >
                          {activityLabels.map(label => (
                            <option key={label} value={label}>{label}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div className="flex items-center gap-1">
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 text-xs px-2 font-bold"
                          onClick={() => applyQuickLabel(quickLabelH)}
                        >
                          <span className="bg-primary/20 text-primary px-1 rounded mr-1">H</span>
                          {quickLabelH}
                        </Button>
                        <select
                          value={quickLabelH}
                          onChange={(e) => setQuickLabelH(e.target.value)}
                          className="h-7 w-6 rounded border border-input bg-background text-xs cursor-pointer"
                          title="H-Taste ändern"
                        >
                          {activityLabels.map(label => (
                            <option key={label} value={label}>{label}</option>
                          ))}
                        </select>
                      </div>
                      
                      <span className="text-xs text-muted-foreground mx-2">|</span>
                      {selectedImages.size > 0 && (
                        <span className="text-xs text-muted-foreground">
                          {selectedImages.size} markiert
                        </span>
                      )}
                      <span className="text-sm text-primary font-medium">
                        {currentImage.detections.length} Erkennungen
                      </span>
                    </div>
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
                          Frame {(currentImage.frameNumber ?? 0)} aus {currentImage.sourceVideo}
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
                        />
                      </div>
                      <Button
                        size="sm"
                        variant="secondary"
                        className="w-full"
                        onClick={handleRedetect}
                        disabled={isProcessing || !model}
                      >
                        Neu erkennen
                      </Button>
                    </div>
                  </div>

                  {/* Batch Labeling */}
                  <div className="pb-4 border-b border-border">
                    <h3 className="font-semibold mb-3 flex items-center gap-2">
                      <Tag className="w-4 h-4" />
                      Batch-Labeling
                    </h3>
                    <div className="space-y-2">
                      <div className="flex gap-2 items-center">
                        <Input
                          type="number"
                          min={1}
                          max={images.length}
                          placeholder="Von"
                          className="text-sm w-16"
                          id="batch-from"
                          defaultValue={1}
                        />
                        <span className="text-muted-foreground">-</span>
                        <Input
                          type="number"
                          min={1}
                          max={images.length}
                          placeholder="Bis"
                          className="text-sm w-16"
                          id="batch-to"
                          defaultValue={images.length}
                        />
                      </div>
                      <select
                        id="batch-label-select"
                        className="w-full h-8 rounded border border-input bg-background text-sm px-2"
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
                            newImages[i].activityLabel = label;

                            if (newImages[i].detections.length === 0 && model) {
                              const detections = await detectObjects(newImages[i].file, confidenceThreshold);
                              // Nur "Person" Detektionen behalten und mit dem Label versehen
                              const personDetections = detections
                                .filter((d) => d.label.toLowerCase() === "person")
                                .map((d) => ({ ...d, label }));
                              newImages[i].detections = personDetections;
                              if (personDetections.length > 0) detectedCount++;
                            } else {
                              // Bestehende Detektionen nur umbenennen
                              newImages[i].detections = newImages[i].detections.map((d) => ({
                                ...d,
                                label: label,
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
                    
                    {/* Bilder markieren und löschen */}
                    <div className="mt-3 pt-3 border-t border-border/50">
                      <p className="text-xs text-muted-foreground mb-2">
                        Bilder markieren & löschen ({selectedImages.size} ausgewählt)
                      </p>
                      <div className="flex gap-2 mb-2">
                        <Button
                          size="sm"
                          variant="outline"
                          className="flex-1 text-xs"
                          onClick={() => {
                            // Aktuelles Bild zur Auswahl hinzufügen/entfernen
                            const newSelection = new Set(selectedImages);
                            if (newSelection.has(currentIndex)) {
                              newSelection.delete(currentIndex);
                            } else {
                              newSelection.add(currentIndex);
                            }
                            setSelectedImages(newSelection);
                          }}
                        >
                          {selectedImages.has(currentIndex) ? "Demarkieren" : "Markieren"}
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="flex-1 text-xs"
                          onClick={() => {
                            // Alle markieren
                            const allSelected = new Set<number>();
                            images.forEach((_, idx) => allSelected.add(idx));
                            setSelectedImages(allSelected);
                          }}
                        >
                          Alle
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="flex-1 text-xs"
                          onClick={() => setSelectedImages(new Set())}
                        >
                          Keine
                        </Button>
                      </div>
                      
                      {/* Bereich markieren */}
                      <div className="flex gap-2 items-center mb-2">
                        <Input
                          type="number"
                          min={1}
                          max={images.length}
                          placeholder="Von"
                          className="text-sm w-16"
                          id="select-from"
                        />
                        <span className="text-muted-foreground text-xs">-</span>
                        <Input
                          type="number"
                          min={1}
                          max={images.length}
                          placeholder="Bis"
                          className="text-sm w-16"
                          id="select-to"
                        />
                        <Button
                          size="sm"
                          variant="outline"
                          className="text-xs"
                          onClick={() => {
                            const fromInput = document.getElementById('select-from') as HTMLInputElement;
                            const toInput = document.getElementById('select-to') as HTMLInputElement;
                            const from = parseInt(fromInput.value) - 1;
                            const to = parseInt(toInput.value) - 1;
                            
                            if (isNaN(from) || isNaN(to) || from < 0 || to >= images.length || from > to) {
                              toast.error(`Ungültiger Bereich (1-${images.length})`);
                              return;
                            }
                            
                            const newSelection = new Set(selectedImages);
                            for (let i = from; i <= to; i++) {
                              newSelection.add(i);
                            }
                            setSelectedImages(newSelection);
                            fromInput.value = '';
                            toInput.value = '';
                            toast.success(`${to - from + 1} Bilder markiert`);
                          }}
                        >
                          Bereich
                        </Button>
                      </div>
                      
                      <Button
                        size="sm"
                        variant="destructive"
                        className="w-full"
                        disabled={selectedImages.size === 0}
                        onClick={() => {
                          if (selectedImages.size === 0) {
                            toast.error("Keine Bilder markiert");
                            return;
                          }
                          
                          const deleteCount = selectedImages.size;
                          const newImages = images.filter((_, idx) => !selectedImages.has(idx));
                          setImages(newImages);
                          
                          // Setze aktuellen Index zurück falls nötig
                          if (currentIndex >= newImages.length) {
                            setCurrentIndex(Math.max(0, newImages.length - 1));
                          }
                          setSelectedDetection(null);
                          setSelectedImages(new Set());
                          
                          toast.success(`${deleteCount} markierte Bilder gelöscht`);
                        }}
                      >
                        <Trash2 className="w-3 h-3 mr-1" />
                        {selectedImages.size} markierte löschen
                      </Button>
                    </div>
                  </div>

                  {/* Export Settings */}
                  <div className="pb-4 border-b border-border">
                    <h3 className="font-semibold mb-3 flex items-center gap-2">
                      <Download className="w-4 h-4" />
                      Export-Einstellungen
                    </h3>
                    
                    <div className="space-y-4">
                      {/* Bounding Box Padding */}
                      <div>
                        <div className="flex justify-between text-sm mb-2">
                          <span className="text-muted-foreground font-medium">Bounding Box Padding:</span>
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
                          Erweitert Erkennungsboxen um {boundingBoxPadding}% links/rechts für Werkzeuge
                        </p>
                      </div>
                      
                      {/* Motion Blur Filter */}
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm text-muted-foreground font-medium">Bewegungsunschärfe filtern:</span>
                          <Button
                            variant={filterMotionBlur ? "default" : "outline"}
                            size="sm"
                            onClick={() => setFilterMotionBlur(!filterMotionBlur)}
                          >
                            {filterMotionBlur ? "Aktiv" : "Aus"}
                          </Button>
                        </div>
                        {filterMotionBlur && (
                          <>
                            <div className="flex items-center gap-3">
                              <span className="text-sm w-10">{blurThreshold}</span>
                              <Slider
                                value={[blurThreshold]}
                                onValueChange={([v]) => setBlurThreshold(v)}
                                min={50}
                                max={300}
                                step={10}
                              />
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">
                              Höher = weniger streng. Niedrig = strenger
                            </p>
                          </>
                        )}
                      </div>
                      
                      {/* Export Only Labeled */}
                      <div>
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm font-medium">Nur gelabelte Frames</p>
                            <p className="text-xs text-muted-foreground">
                              Ungelabelte werden ignoriert
                            </p>
                          </div>
                          <Button
                            variant={exportOnlyLabeled ? "default" : "outline"}
                            size="sm"
                            onClick={() => setExportOnlyLabeled(!exportOnlyLabeled)}
                          >
                            {exportOnlyLabeled ? "Aktiv" : "Aus"}
                          </Button>
                        </div>
                        
                        <div className="text-xs text-muted-foreground mt-2 p-2 bg-secondary/50 rounded space-y-1">
                          <p className="font-medium">Export-Struktur:</p>
                          <div className="pl-2 font-mono">
                            <p>Label/</p>
                            <p className="pl-3">└─ Label01/</p>
                            <p className="pl-6">├─ frames00001.jpg</p>
                            <p className="pl-6">└─ frames00002.jpg</p>
                            <p>Label labeld/</p>
                            <p className="pl-3">└─ Label01/</p>
                            <p className="pl-6">├─ frames00001.json</p>
                            <p className="pl-6">└─ frames00002.json</p>
                          </div>
                          <p className="mt-2 text-xs italic">Jedes Label beginnt bei frames00001</p>
                          <p className="text-xs italic">Clip-Ordner fortlaufend über Videos</p>
                        </div>
                        
                        {/* Clip-Zähler Anzeige & Reset */}
                        <div className="mt-3 p-2 bg-primary/10 rounded border border-primary/20">
                          <div className="flex items-center justify-between mb-2">
                            <p className="text-xs font-medium">Aktuelle Clip-Zähler:</p>
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-6 text-xs"
                              onClick={() => {
                                setLabelClipCounters({});
                                toast.success("Alle Zähler zurückgesetzt für neues Projekt!");
                              }}
                            >
                              <RotateCcw className="w-3 h-3 mr-1" />
                              Neu starten
                            </Button>
                          </div>
                          {Object.keys(labelClipCounters).length > 0 ? (
                            <div className="space-y-1">
                              {Object.entries(labelClipCounters).map(([label, count]) => (
                                <div key={label} className="flex justify-between text-xs">
                                  <span className="text-muted-foreground">{label}:</span>
                                  <span className="font-mono">{count} Clips → nächster: {String(count + 1).padStart(2, "0")}</span>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <p className="text-xs text-muted-foreground italic">Noch keine Exports - startet bei 01</p>
                          )}
                        </div>
                      </div>
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
                              newImages[currentIndex].activityLabel = label;
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
                                      newImages[currentIndex].activityLabel = editLabelValue;
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
                                    newImages[currentIndex].activityLabel = editLabelValue;
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
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};
