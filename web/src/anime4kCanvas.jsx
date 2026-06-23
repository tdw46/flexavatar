import React from "react";
import Anime4K from "anime4k";

const TARGET_SCALE = 2;

function targetSizeFor(width, height) {
  return {
    width: Math.max(1, Math.round(width * TARGET_SCALE)),
    height: Math.max(1, Math.round(height * TARGET_SCALE)),
    scale: TARGET_SCALE,
  };
}

function resizeCanvas(canvas, width, height) {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function drawSourceFrame(image, sourceCanvas) {
  if (!image.naturalWidth || !image.naturalHeight) return false;
  resizeCanvas(sourceCanvas, image.naturalWidth, image.naturalHeight);
  const context = sourceCanvas.getContext("2d", { alpha: true, willReadFrequently: false });
  if (!context) return false;
  context.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
  context.drawImage(image, 0, 0, sourceCanvas.width, sourceCanvas.height);
  sourceCanvas.videoWidth = sourceCanvas.width;
  sourceCanvas.videoHeight = sourceCanvas.height;
  return true;
}

function renderAnime4KFast(upscaledCanvas, sourceCanvas) {
  const gl = upscaledCanvas.getContext("webgl", {
    alpha: true,
    antialias: false,
    preserveDrawingBuffer: false,
  });
  if (!gl) throw new Error("WebGL is not available for Anime4K Fast");

  const scaler = Anime4K.Scaler(gl);
  scaler.inputVideo(sourceCanvas);
  const { width, height, scale } = targetSizeFor(sourceCanvas.width, sourceCanvas.height);
  scaler.setSize(scale);
  resizeCanvas(upscaledCanvas, width, height);

  let frame = 0;
  const tick = () => {
    try {
      sourceCanvas.videoWidth = sourceCanvas.width;
      sourceCanvas.videoHeight = sourceCanvas.height;
      scaler.render();
    } catch {
      // The multipart source can be between decoded frames; keep the last good canvas.
    }
    frame = window.requestAnimationFrame(tick);
  };
  frame = window.requestAnimationFrame(tick);
  return () => window.cancelAnimationFrame(frame);
}

function drawCompositedFrame(canvas, imageCanvas, camera, outputSize) {
  if (!imageCanvas.width || !imageCanvas.height) return false;
  const width = outputSize?.width ?? 1600;
  const height = outputSize?.height ?? 1600;
  resizeCanvas(canvas, width, height);

  const context = canvas.getContext("2d", { alpha: true, willReadFrequently: false });
  if (!context) return false;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "rgb(248, 252, 252)";
  context.fillRect(0, 0, width, height);

  const radius = Math.max(0.35, Math.min(1.8, Number(camera?.radius ?? 1)));
  const scale = 0.86 / radius;
  const maxFill = 0.98;
  const targetW = Math.max(96, Math.round(width * Math.min(maxFill, scale)));
  const targetH = Math.max(96, Math.round(height * Math.min(maxFill, scale)));
  const containScale = Math.min(targetW / imageCanvas.width, targetH / imageCanvas.height);
  const drawW = Math.max(1, Math.round(imageCanvas.width * containScale));
  const drawH = Math.max(1, Math.round(imageCanvas.height * containScale));
  const yaw = Number(camera?.yaw ?? 0);
  const pitch = Number(camera?.pitch ?? 0);
  const x = Math.round((width - drawW) / 2 + Math.max(-1, Math.min(1, yaw / 40)) * width * 0.035);
  const y = Math.round((height - drawH) / 2 - Math.max(-1, Math.min(1, pitch / 40)) * height * 0.028);

  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(imageCanvas, x, y, drawW, drawH);
  return true;
}

export function Anime4KCanvas({ src, alt, enabled, camera, outputSize }) {
  const imageRef = React.useRef(null);
  const canvasRef = React.useRef(null);
  const sourceCanvasRef = React.useRef(document.createElement("canvas"));
  const upscaledCanvasRef = React.useRef(document.createElement("canvas"));
  const [status, setStatus] = React.useState("");

  React.useEffect(() => {
    if (!enabled) return undefined;
    let frame = 0;
    const tick = () => {
      const image = imageRef.current;
      const sourceCanvas = sourceCanvasRef.current;
      if (image && sourceCanvas) {
        drawSourceFrame(image, sourceCanvas);
      }
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [enabled, src]);

  React.useEffect(() => {
    if (!enabled) return undefined;
    let frame = 0;
    const tick = () => {
      const canvas = canvasRef.current;
      const upscaledCanvas = upscaledCanvasRef.current;
      if (canvas && upscaledCanvas) {
        drawCompositedFrame(canvas, upscaledCanvas, camera, outputSize);
      }
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [camera, enabled, outputSize]);

  React.useEffect(() => {
    if (!enabled) return undefined;

    let cleanup = null;
    let startFrame = 0;
    let cancelled = false;
    const start = () => {
      const image = imageRef.current;
      const canvas = canvasRef.current;
      const sourceCanvas = sourceCanvasRef.current;
      if (!image || !canvas || !sourceCanvas || !drawSourceFrame(image, sourceCanvas)) {
        startFrame = window.requestAnimationFrame(start);
        return;
      }

      try {
        cleanup = renderAnime4KFast(upscaledCanvasRef.current, sourceCanvas);
        if (!cancelled) setStatus("");
      } catch (error) {
        if (cancelled) return;
        setStatus(error instanceof Error ? error.message : "Anime4K Fast failed");
      }
    };

    startFrame = window.requestAnimationFrame(start);
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(startFrame);
      if (cleanup) cleanup();
    };
  }, [enabled, src]);

  if (!enabled) {
    return <img className="renderStream" src={src} alt={alt} />;
  }

  return (
    <div className="anime4kStage">
      <img ref={imageRef} className="anime4kSource" src={src} alt={alt} crossOrigin="anonymous" />
      <canvas ref={canvasRef} className="renderStream anime4kCanvas" aria-label={alt} />
      {status ? <span className="anime4kBadge">{status}</span> : null}
    </div>
  );
}
