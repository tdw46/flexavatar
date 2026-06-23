import React from "react";
import Anime4K from "anime4k";
import * as Anime4KWebGPU from "anime4k-webgpu";

const TARGET_SCALE = 2;
const ANIME4K_OPTIONS = { blur: 4.0, bold: 7.5 };
const THA4_DISPLAY_PADDING = 26;

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
  const context = sourceCanvas.getContext("2d", { alpha: true, willReadFrequently: true });
  if (!context) return false;
  context.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
  context.drawImage(image, 0, 0, sourceCanvas.width, sourceCanvas.height);
  sourceCanvas.videoWidth = sourceCanvas.width;
  sourceCanvas.videoHeight = sourceCanvas.height;
  sourceCanvas.contentBounds = findAlphaBounds(context, sourceCanvas.width, sourceCanvas.height);
  return true;
}

function findAlphaBounds(context, width, height) {
  const data = context.getImageData(0, 0, width, height).data;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (data[(y * width + x) * 4 + 3] <= 0) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }
  if (maxX < minX || maxY < minY) {
    return { x: 0, y: 0, width, height };
  }
  const padding = THA4_DISPLAY_PADDING;
  const x = Math.max(0, minX - padding);
  const y = Math.max(0, minY - padding);
  const right = Math.min(width, maxX + 1 + padding);
  const bottom = Math.min(height, maxY + 1 + padding);
  return { x, y, width: right - x, height: bottom - y };
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
  resizeCanvas(upscaledCanvas, width, height);
  scaler.resize(scale, ANIME4K_OPTIONS);

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

async function renderAnime4KWebGpu(upscaledCanvas, sourceCanvas) {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available");
  }

  const { width, height } = targetSizeFor(sourceCanvas.width, sourceCanvas.height);
  resizeCanvas(upscaledCanvas, width, height);

  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.autoplay = true;

  const stream = sourceCanvas.captureStream(30);
  video.srcObject = stream;
  await video.play();

  await new Promise((resolve) => {
    if (video.videoWidth && video.videoHeight) {
      resolve();
      return;
    }
    video.onloadedmetadata = () => resolve();
  });

  await Anime4KWebGPU.render({
    video,
    canvas: upscaledCanvas,
    pipelineBuilder: (device, inputTexture) => {
      const upscale = new Anime4KWebGPU.CNNx2M({
        device,
        inputTexture,
      });
      return [upscale];
    },
  });

  return () => {
    video.pause();
    stream.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
  };
}

async function startAnime4KRenderer(upscaledCanvas, sourceCanvas) {
  try {
    return await renderAnime4KWebGpu(upscaledCanvas, sourceCanvas);
  } catch (error) {
    console.warn("Anime4K WebGPU unavailable, using WebGL fallback.", error);
    return renderAnime4KFast(upscaledCanvas, sourceCanvas);
  }
}

function drawCompositedFrame(canvas, imageCanvas, sourceBounds, camera, outputSize) {
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
  const sourceScaleX = imageCanvas.width / Math.max(1, sourceBounds?.sourceWidth ?? imageCanvas.width / TARGET_SCALE);
  const sourceScaleY = imageCanvas.height / Math.max(1, sourceBounds?.sourceHeight ?? imageCanvas.height / TARGET_SCALE);
  const sx = Math.round((sourceBounds?.x ?? 0) * sourceScaleX);
  const sy = Math.round((sourceBounds?.y ?? 0) * sourceScaleY);
  const sw = Math.max(1, Math.round((sourceBounds?.width ?? imageCanvas.width / sourceScaleX) * sourceScaleX));
  const sh = Math.max(1, Math.round((sourceBounds?.height ?? imageCanvas.height / sourceScaleY) * sourceScaleY));
  const containScale = Math.min(targetW / sw, targetH / sh);
  const drawW = Math.max(1, Math.round(sw * containScale));
  const drawH = Math.max(1, Math.round(sh * containScale));
  const yaw = Number(camera?.yaw ?? 0);
  const pitch = Number(camera?.pitch ?? 0);
  const x = Math.round((width - drawW) / 2 + Math.max(-1, Math.min(1, yaw / 40)) * width * 0.035);
  const y = Math.round((height - drawH) / 2 - Math.max(-1, Math.min(1, pitch / 40)) * height * 0.028);

  context.imageSmoothingEnabled = false;
  context.drawImage(imageCanvas, sx, sy, sw, sh, x, y, drawW, drawH);
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
      const sourceCanvas = sourceCanvasRef.current;
      if (canvas && upscaledCanvas && sourceCanvas) {
        const bounds = sourceCanvas.contentBounds
          ? { ...sourceCanvas.contentBounds, sourceWidth: sourceCanvas.width, sourceHeight: sourceCanvas.height }
          : null;
        drawCompositedFrame(canvas, upscaledCanvas, bounds, camera, outputSize);
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
        Promise.resolve(startAnime4KRenderer(upscaledCanvasRef.current, sourceCanvas))
          .then((nextCleanup) => {
            if (cancelled) {
              nextCleanup();
              return;
            }
            cleanup = nextCleanup;
            setStatus("");
          })
          .catch((error) => {
            if (cancelled) return;
            setStatus(error instanceof Error ? error.message : "Anime4K failed");
          });
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
