import React from "react";
import Anime4K from "anime4k";
import * as Anime4KWebGPU from "anime4k-webgpu";

const TARGET_SCALE = 4;
const ANIME4K_OPTIONS = { blur: 2.0, bold: 6.0 };
const THA4_DISPLAY_PADDING = 26;
const STAGE_BACKGROUND = "rgb(248, 252, 252)";
const MODEL_SOURCE_MAX = 512;
const REAL_CUGAN_SOURCE_MAX = 512;
const MODEL_IDLE_DELAY_MS = 120;
const MODEL_BUSY_POLL_MS = 16;
const MODEL_UPSCALER_MODES = new Set(["artcnn", "artcnnDs", "animejanai", "realcugan"]);
const MODEL_UPSCALER_LABELS = {
  artcnn: "ArtCNN C4F32",
  artcnnDs: "ArtCNN C4F32 DS",
  animejanai: "AnimeJaNai V3.1 Sharp",
  realcugan: "Real-CUGAN",
};
const WEBGPU_UPSCALER_MODES = new Set([
  "modeA",
  "modeB",
  "modeC",
  "ultraDetail",
  "anime4k-modeAA",
  "anime4k-modeB",
  "anime4k-modeBB",
  "anime4k-modeCA",
  "anime4k-clean",
  "anime4k-restore",
  "anime4k-denoise",
  "anime4k-gan4x",
  "anime4k-modeA",
  "anime4k-modeC",
  "anime4k-ultraDetail",
]);

let modelWorker = null;
let modelWorkerRequestId = 0;

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

function drawSourceFrame(image, sourceCanvas, { forceBounds = false } = {}) {
  if (!image.naturalWidth || !image.naturalHeight) return false;
  resizeCanvas(sourceCanvas, image.naturalWidth, image.naturalHeight);
  const context = sourceCanvas.getContext("2d", { alpha: true, willReadFrequently: true });
  if (!context) return false;
  context.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
  context.drawImage(image, 0, 0, sourceCanvas.width, sourceCanvas.height);
  sourceCanvas.videoWidth = sourceCanvas.width;
  sourceCanvas.videoHeight = sourceCanvas.height;
  const boundsKey = `${sourceCanvas.width}x${sourceCanvas.height}`;
  const now = performance.now();
  if (forceBounds || sourceCanvas.boundsKey !== boundsKey || !sourceCanvas.contentBounds || now - (sourceCanvas.lastBoundsAt ?? 0) > 1000) {
    sourceCanvas.contentBounds = findAlphaBounds(context, sourceCanvas.width, sourceCanvas.height);
    sourceCanvas.boundsKey = boundsKey;
    sourceCanvas.lastBoundsAt = now;
  }
  context.globalCompositeOperation = "destination-over";
  context.fillStyle = STAGE_BACKGROUND;
  context.fillRect(0, 0, sourceCanvas.width, sourceCanvas.height);
  context.globalCompositeOperation = "source-over";
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

function sourceImageData(sourceCanvas) {
  const context = sourceCanvas.getContext("2d", { alpha: true, willReadFrequently: true });
  if (!context) return null;
  return context.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
}

function imageDataSignature(imageData) {
  const { data, width, height } = imageData;
  let hash = 2166136261;
  for (let index = 0; index < data.length; index += 1) {
    hash ^= data[index];
    hash = Math.imul(hash, 16777619);
  }
  return `${width}x${height}:${hash >>> 0}`;
}

function scaledSourceCanvas(sourceCanvas, maxSize) {
  const longestSide = Math.max(sourceCanvas.width, sourceCanvas.height);
  if (longestSide <= maxSize) return sourceCanvas;

  const scale = maxSize / longestSide;
  const canvas = document.createElement("canvas");
  resizeCanvas(canvas, Math.max(1, Math.round(sourceCanvas.width * scale)), Math.max(1, Math.round(sourceCanvas.height * scale)));
  const context = canvas.getContext("2d", { alpha: true, willReadFrequently: false });
  if (!context) return sourceCanvas;
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(sourceCanvas, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function copySourceToCanvas(canvas, sourceCanvas) {
  const context = canvas.getContext("2d", { alpha: true, willReadFrequently: false });
  if (!context) return;
  resizeCanvas(canvas, sourceCanvas.width, sourceCanvas.height);
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(sourceCanvas, 0, 0);
}

function getModelWorker() {
  if (!modelWorker) {
    modelWorker = new Worker(new URL("./upscalerWorker.js", import.meta.url), { type: "module" });
  }
  return modelWorker;
}

function writeWorkerResult(canvas, result) {
  resizeCanvas(canvas, result.width, result.height);
  const context = canvas.getContext("2d", { alpha: true, willReadFrequently: false });
  if (!context) return;
  const image = new ImageData(new Uint8ClampedArray(result.data), result.width, result.height);
  context.putImageData(image, 0, 0);
}

function modelSourceSnapshot(mode, sourceCanvas) {
  const maxSize = mode === "realcugan" ? REAL_CUGAN_SOURCE_MAX : MODEL_SOURCE_MAX;
  const inferenceCanvas = scaledSourceCanvas(sourceCanvas, maxSize);
  const imageData = sourceImageData(inferenceCanvas);
  if (!imageData) return null;
  return {
    signature: imageDataSignature(imageData),
    imageData,
  };
}

function runWorkerUpscale(mode, snapshot, shouldContinue) {
  if (!snapshot?.imageData) return Promise.resolve(null);
  const { imageData, signature } = snapshot;
  const worker = getModelWorker();
  const id = (modelWorkerRequestId += 1);
  return new Promise((resolve, reject) => {
    const handleMessage = (event) => {
      if (event.data?.id !== id) return;
      worker.removeEventListener("message", handleMessage);
      if (!shouldContinue()) {
        resolve(null);
        return;
      }
      if (!event.data.ok) {
        reject(new Error(event.data.error ?? "Upscaler worker failed"));
        return;
      }
      resolve({
        signature,
        width: event.data.width,
        height: event.data.height,
        data: event.data.data,
      });
    };

    worker.addEventListener("message", handleMessage);
    worker.postMessage(
      {
        id,
        mode,
        width: imageData.width,
        height: imageData.height,
        data: imageData.data.buffer,
      },
      [imageData.data.buffer],
    );
  });
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

function makePresetPipeline(mode, device, inputTexture, video, upscaledCanvas) {
  const descriptor = {
    device,
    inputTexture,
    nativeDimensions: {
      width: video.videoWidth,
      height: video.videoHeight,
    },
    targetDimensions: {
      width: upscaledCanvas.width,
      height: upscaledCanvas.height,
    },
  };

  const denoise = (texture, intensity = 0.14, spatial = 1.6) => {
    const pipeline = new Anime4KWebGPU.BilateralMean({ device, inputTexture: texture });
    pipeline.updateParam("strength", intensity);
    pipeline.updateParam("strength2", spatial);
    return pipeline;
  };

  const restore = (texture, soft = false) =>
    soft
      ? new Anime4KWebGPU.CNNSoftM({ device, inputTexture: texture })
      : new Anime4KWebGPU.CNNM({ device, inputTexture: texture });

  const presetWithClamp = (Preset) => {
    const clamp = new Anime4KWebGPU.ClampHighlights({ device, inputTexture });
    const preset = new Preset({
      ...descriptor,
      inputTexture: clamp.getOutputTexture(),
    });
    return [clamp, preset];
  };

  if (mode === "modeA" || mode === "anime4k-modeA") {
    return presetWithClamp(Anime4KWebGPU.ModeA);
  }

  if (mode === "modeB" || mode === "anime4k-modeB") {
    return presetWithClamp(Anime4KWebGPU.ModeB);
  }

  if (mode === "modeC" || mode === "anime4k-modeC") {
    return presetWithClamp(Anime4KWebGPU.ModeC);
  }

  if (mode === "anime4k-modeAA") {
    return presetWithClamp(Anime4KWebGPU.ModeAA);
  }

  if (mode === "anime4k-modeBB") {
    return presetWithClamp(Anime4KWebGPU.ModeBB);
  }

  if (mode === "anime4k-modeCA") {
    return presetWithClamp(Anime4KWebGPU.ModeCA);
  }

  if (mode === "anime4k-clean") {
    const clamp = new Anime4KWebGPU.ClampHighlights({ device, inputTexture });
    const firstDenoise = denoise(clamp.getOutputTexture());
    const firstRestore = restore(firstDenoise.getOutputTexture(), true);
    const firstUpscale = new Anime4KWebGPU.DenoiseCNNx2VL({
      device,
      inputTexture: firstRestore.getOutputTexture(),
    });
    const secondUpscale = new Anime4KWebGPU.CNNx2M({
      device,
      inputTexture: firstUpscale.getOutputTexture(),
    });
    return [clamp, firstDenoise, firstRestore, firstUpscale, secondUpscale];
  }

  if (mode === "anime4k-restore") {
    const clamp = new Anime4KWebGPU.ClampHighlights({ device, inputTexture });
    const firstRestore = restore(clamp.getOutputTexture());
    const firstUpscale = new Anime4KWebGPU.CNNx2M({
      device,
      inputTexture: firstRestore.getOutputTexture(),
    });
    const secondRestore = restore(firstUpscale.getOutputTexture());
    const secondUpscale = new Anime4KWebGPU.CNNx2M({
      device,
      inputTexture: secondRestore.getOutputTexture(),
    });
    return [clamp, firstRestore, firstUpscale, secondRestore, secondUpscale];
  }

  if (mode === "anime4k-denoise") {
    const clamp = new Anime4KWebGPU.ClampHighlights({ device, inputTexture });
    const firstUpscale = new Anime4KWebGPU.DenoiseCNNx2VL({ device, inputTexture: clamp.getOutputTexture() });
    const secondUpscale = new Anime4KWebGPU.DenoiseCNNx2VL({
      device,
      inputTexture: firstUpscale.getOutputTexture(),
    });
    return [clamp, firstUpscale, secondUpscale];
  }

  if (mode === "anime4k-gan4x") {
    const upscale = new Anime4KWebGPU.GANx4UUL({
      device,
      inputTexture,
    });
    const finalRestore = new Anime4KWebGPU.GANUUL({
      device,
      inputTexture: upscale.getOutputTexture(),
    });
    return [upscale, finalRestore];
  }

  if (mode === "ultraDetail" || mode === "anime4k-ultraDetail") {
    const clamp = new Anime4KWebGPU.ClampHighlights({
      device,
      inputTexture,
    });
    const firstUpscale = new Anime4KWebGPU.CNNx2UL({
      device,
      inputTexture: clamp.getOutputTexture(),
    });
    const secondUpscale = new Anime4KWebGPU.CNNx2M({
      device,
      inputTexture: firstUpscale.getOutputTexture(),
    });
    return [clamp, firstUpscale, secondUpscale];
  }

  return presetWithClamp(Anime4KWebGPU.ModeBB);
}

function isAnime4KMode(mode) {
  return WEBGPU_UPSCALER_MODES.has(mode);
}

async function renderAnime4KWebGpu(upscaledCanvas, sourceCanvas, mode) {
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
    pipelineBuilder: (device, inputTexture) => makePresetPipeline(mode, device, inputTexture, video, upscaledCanvas),
  });

  return () => {
    video.pause();
    stream.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
  };
}

async function startAnime4KRenderer(upscaledCanvas, sourceCanvas, mode) {
  try {
    return await renderAnime4KWebGpu(upscaledCanvas, sourceCanvas, mode);
  } catch (error) {
    console.warn("Anime4K WebGPU unavailable, using WebGL fallback.", error);
    return renderAnime4KFast(upscaledCanvas, sourceCanvas);
  }
}

function startModelRenderer(upscaledCanvas, sourceCanvas, mode, setStatus, setModelBusy) {
  let timer = 0;
  let previewFrame = 0;
  let stopped = false;
  let busy = false;
  let renderedOnce = false;
  let activeSignature = "";
  let renderedSignature = "";
  const shouldContinue = () => !stopped;
  copySourceToCanvas(upscaledCanvas, sourceCanvas);
  setStatus(`Rendering ${MODEL_UPSCALER_LABELS[mode] ?? "model"}...`);
  setModelBusy(false);

  const previewRawWhileBusy = () => {
    if (stopped) return;
    if (busy) {
      copySourceToCanvas(upscaledCanvas, sourceCanvas);
    }
    previewFrame = window.requestAnimationFrame(previewRawWhileBusy);
  };

  const schedule = (delay = MODEL_BUSY_POLL_MS) => {
    if (!stopped) {
      timer = window.setTimeout(tick, delay);
    }
  };

  const tick = () => {
    if (stopped) return;
    if (busy) {
      schedule();
      return;
    }
    const snapshot = modelSourceSnapshot(mode, sourceCanvas);
    if (!snapshot) {
      schedule(MODEL_BUSY_POLL_MS);
      return;
    }
    if (snapshot.signature === activeSignature || snapshot.signature === renderedSignature) {
      schedule(MODEL_IDLE_DELAY_MS);
      return;
    }
    activeSignature = snapshot.signature;
    busy = true;
    setModelBusy(true);
    copySourceToCanvas(upscaledCanvas, sourceCanvas);
    Promise.resolve(runWorkerUpscale(mode, snapshot, shouldContinue))
      .then((result) => {
        if (!shouldContinue() || !result) return;
        if (result.signature !== activeSignature) return;
        const latestSignature = modelSourceSnapshot(mode, sourceCanvas)?.signature;
        if (latestSignature && latestSignature !== result.signature) return;
        writeWorkerResult(upscaledCanvas, result);
        renderedSignature = result.signature;
        renderedOnce = true;
        setStatus("");
      })
      .catch((error) => {
        if (shouldContinue()) setStatus(error instanceof Error ? error.message : "Model upscaler failed");
      })
      .finally(() => {
        busy = false;
        if (shouldContinue()) setModelBusy(false);
        activeSignature = "";
        if (shouldContinue()) schedule(renderedOnce ? MODEL_IDLE_DELAY_MS : MODEL_BUSY_POLL_MS);
      });
  };

  schedule(MODEL_IDLE_DELAY_MS);
  previewFrame = window.requestAnimationFrame(previewRawWhileBusy);
  return () => {
    stopped = true;
    setModelBusy(false);
    window.clearTimeout(timer);
    window.cancelAnimationFrame(previewFrame);
  };
}

function drawCompositedFrame(canvas, imageCanvas, sourceBounds, camera, outputSize) {
  if (!imageCanvas.width || !imageCanvas.height) return false;
  const width = outputSize?.width ?? 1600;
  const height = outputSize?.height ?? 1600;
  resizeCanvas(canvas, width, height);

  const context = canvas.getContext("2d", { alpha: true, willReadFrequently: false });
  if (!context) return false;
  context.clearRect(0, 0, width, height);
  context.fillStyle = STAGE_BACKGROUND;
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

export function Anime4KCanvas({ src, alt, enabled, mode = "modeA", camera, outputSize }) {
  const imageRef = React.useRef(null);
  const canvasRef = React.useRef(null);
  const sourceCanvasRef = React.useRef(document.createElement("canvas"));
  const upscaledCanvasRef = React.useRef(document.createElement("canvas"));
  const [status, setStatus] = React.useState("");
  const [modelBusy, setModelBusy] = React.useState(false);

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
      if (!image || !canvas || !sourceCanvas || !drawSourceFrame(image, sourceCanvas, { forceBounds: true })) {
        startFrame = window.requestAnimationFrame(start);
        return;
      }

      try {
        const upscaledCanvas = document.createElement("canvas");
        upscaledCanvasRef.current = upscaledCanvas;
        const renderer = isAnime4KMode(mode)
          ? startAnime4KRenderer(upscaledCanvas, sourceCanvas, mode)
          : startModelRenderer(upscaledCanvas, sourceCanvas, mode, setStatus, setModelBusy);

        Promise.resolve(renderer)
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
            setStatus(error instanceof Error ? error.message : "Upscaler failed");
          });
      } catch (error) {
        if (cancelled) return;
        setStatus(error instanceof Error ? error.message : "Upscaler failed");
      }
    };

    startFrame = window.requestAnimationFrame(start);
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(startFrame);
      if (cleanup) cleanup();
    };
  }, [enabled, mode, src]);

  if (!enabled) {
    return <img className="renderStream" src={src} alt={alt} />;
  }

  return (
    <div className="anime4kStage">
      <img ref={imageRef} className="anime4kSource" src={src} alt={alt} crossOrigin="anonymous" />
      <canvas ref={canvasRef} className="renderStream anime4kCanvas" aria-label={alt} />
      {modelBusy ? <span className="upscalerSpinner" aria-label="Upscaler rendering" /> : null}
      {status ? <span className="anime4kBadge">{status}</span> : null}
    </div>
  );
}
