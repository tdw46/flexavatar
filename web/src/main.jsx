import React from "react";
import { createRoot } from "react-dom/client";
import {
  Aperture,
  Camera,
  Check,
  ChevronRight,
  Download,
  ImagePlus,
  Loader2,
  Play,
  Radio,
  SlidersHorizontal,
  Sparkles,
  Square,
  Upload,
  Video,
} from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

const steps = [
  { id: "input", label: "Input", icon: ImagePlus },
  { id: "generate", label: "Generate", icon: Sparkles },
  { id: "preview", label: "Preview", icon: SlidersHorizontal },
  { id: "webcam", label: "Webcam", icon: Video },
];

const expressionControls = [
  "Brow lift",
  "Eye focus",
  "Cheek raise",
  "Smile",
  "Mouth open",
  "Lip narrow",
  "Nose flare",
  "Squint",
  "Puff",
  "Frown",
  "Jaw bias",
  "Micro asym",
];

function cx(...classes) {
  return classes.filter(Boolean).join(" ");
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json();
}

function App() {
  const [activeStep, setActiveStep] = React.useState("input");
  const [state, setState] = React.useState(null);
  const [avatars, setAvatars] = React.useState([]);
  const [selectedInput, setSelectedInput] = React.useState(null);
  const [selectedPreview, setSelectedPreview] = React.useState(null);
  const [loadedAvatar, setLoadedAvatar] = React.useState(null);
  const [generationPending, setGenerationPending] = React.useState(false);
  const [controls, setControls] = React.useState({
    mode: "default",
    playing: true,
    lockHead: false,
    expression: Array(12).fill(0),
    jaw: [0, 0, 0],
    head: [0, 0, 0],
    camera: { yaw: 0, pitch: 0, roll: 0, radius: 1 },
  });
  const [webcamOn, setWebcamOn] = React.useState(false);
  const [exportedPly, setExportedPly] = React.useState(null);
  const [notice, setNotice] = React.useState("");
  const videoRef = React.useRef(null);
  const driveTimer = React.useRef(null);

  const pushControls = React.useCallback(async (nextControls) => {
    try {
      const result = await api("/api/controls", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: nextControls.mode,
          playing: nextControls.playing,
          lock_head: nextControls.lockHead,
          expression: nextControls.expression,
          jaw: nextControls.jaw,
          head: nextControls.head,
          camera: nextControls.camera,
        }),
      });
      if (result.state) {
        setState(result.state);
      }
    } catch (error) {
      setNotice(error.message);
    }
  }, []);

  const stopWebcam = React.useCallback(() => {
    if (driveTimer.current) {
      window.clearInterval(driveTimer.current);
      driveTimer.current = null;
    }
    const stream = videoRef.current?.srcObject;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
  }, []);

  const sendWebcamFrame = React.useCallback(async () => {
    if (!videoRef.current || videoRef.current.readyState < 2) return;
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");
    const video = videoRef.current;
    const side = Math.min(video.videoWidth, video.videoHeight);
    const sx = (video.videoWidth - side) / 2;
    const sy = (video.videoHeight - side) / 2;
    ctx.drawImage(video, sx, sy, side, side, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const formData = new FormData();
      formData.append("file", blob, "webcam.jpg");
      try {
        await api("/api/drive-frame", { method: "POST", body: formData });
      } catch (error) {
        setNotice(error.message);
      }
    }, "image/jpeg", 0.78);
  }, []);

  const refreshState = React.useCallback(async () => {
    try {
      const nextState = await api("/api/state");
      setState(nextState);
    } catch (error) {
      setNotice(error.message);
    }
  }, []);

  const refreshAvatars = React.useCallback(async () => {
    try {
      const result = await api("/api/avatars");
      setAvatars(result.avatars);
    } catch (error) {
      setNotice(error.message);
    }
  }, []);

  React.useEffect(() => {
    refreshState();
    refreshAvatars();
    const interval = window.setInterval(refreshState, 1000);
    return () => window.clearInterval(interval);
  }, [refreshAvatars, refreshState]);

  React.useEffect(() => {
    if (!loadedAvatar) return;
    pushControls(controls);
  }, [controls, loadedAvatar, pushControls]);

  React.useEffect(() => {
    if (!generationPending || !state?.lastError) return;
    setGenerationPending(false);
    setNotice(`Generation failed: ${state.lastError}`);
    setActiveStep("generate");
  }, [generationPending, state?.lastError]);

  React.useEffect(() => {
    if (!webcamOn) {
      stopWebcam();
      return undefined;
    }

    let stream;
    navigator.mediaDevices
      .getUserMedia({ video: { width: 960, height: 960 }, audio: false })
      .then((nextStream) => {
        stream = nextStream;
        if (videoRef.current) {
          videoRef.current.srcObject = nextStream;
          videoRef.current.play();
        }
        driveTimer.current = window.setInterval(sendWebcamFrame, 220);
      })
      .catch((error) => setNotice(error.message));

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      if (driveTimer.current) {
        window.clearInterval(driveTimer.current);
      }
    };
  }, [sendWebcamFrame, stopWebcam, webcamOn]);

  async function onFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    setNotice("Uploading input...");
    const result = await api("/api/upload", { method: "POST", body: formData });
    setSelectedInput(result);
    setSelectedPreview(`${API_BASE}${result.url}?t=${Date.now()}`);
    setLoadedAvatar(null);
    setExportedPly(null);
    setNotice(`Selected ${result.avatarName}`);
    setActiveStep("generate");
  }

  async function waitForGeneration(avatarName) {
    for (let attempt = 0; attempt < 900; attempt += 1) {
      const nextState = await api("/api/state");
      setState(nextState);
      if (nextState.lastError) {
        throw new Error(nextState.lastError);
      }
      if (!nextState.busy && nextState.lastGeneratedAvatar === avatarName) {
        return nextState;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 500));
    }
    throw new Error("Generation did not finish within 30 minutes.");
  }

  async function exportPly() {
    const result = await api("/api/export-ply", { method: "POST" });
    setExportedPly(`${API_BASE}${result.url}`);
    setNotice("PLY exported for web splat viewers.");
    return result;
  }

  async function generateSelected() {
    if (!selectedInput) return;
    setState((current) =>
      current
        ? {
            ...current,
            busy: true,
            status: "Generation started.",
            lastError: null,
            lastGeneratedAvatar: null,
          }
        : current,
    );
    setGenerationPending(true);
    setLoadedAvatar(null);
    setExportedPly(null);
    setNotice("Generation started.");
    try {
      await api("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: selectedInput.path }),
      });
      await waitForGeneration(selectedInput.avatarName);
      setLoadedAvatar(selectedInput.avatarName);
      setNotice(`Generated ${selectedInput.avatarName}. Opening the Three.js splat viewport.`);
      await exportPly();
      await refreshAvatars();
      setActiveStep("preview");
    } catch (error) {
      setNotice(`Generation failed: ${error.message}`);
      setActiveStep("generate");
    } finally {
      setGenerationPending(false);
      refreshState();
    }
  }

  async function loadAvatar(name) {
    await api(`/api/load/${name}`, { method: "POST" });
    setLoadedAvatar(name);
    setSelectedInput(null);
    setSelectedPreview(null);
    setNotice(`Loaded ${name}. Opening the Three.js splat viewport.`);
    await exportPly();
    setActiveStep("preview");
    refreshState();
  }

  const isBusy = state?.busy || generationPending;
  const hasAvatar = Boolean(loadedAvatar);
  const displayAvatar = loadedAvatar ?? selectedInput?.avatarName ?? "No avatar loaded";
  const currentIndex = steps.findIndex((step) => step.id === activeStep);

  return (
    <main className="studio">
      <aside className="rail">
        <div className="brand">
          <span className="brandMark"><Aperture size={22} /></span>
          <div>
            <strong>FlexAvatar</strong>
            <small>Web Studio</small>
          </div>
        </div>

        <nav className="steps">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const disabled = (step.id === "preview" || step.id === "webcam") && !hasAvatar;
            return (
              <button
                key={step.id}
                className={cx("stepButton", activeStep === step.id && "active", index < currentIndex && "complete", disabled && "disabled")}
                aria-disabled={disabled}
                onClick={() => {
                  if (!disabled) setActiveStep(step.id);
                }}
              >
                <span><Icon size={18} /></span>
                <strong>{step.label}</strong>
                {index < currentIndex ? <Check size={16} /> : <ChevronRight size={16} />}
              </button>
            );
          })}
        </nav>

        <div className="railFooter">
          <span className={cx("statusDot", isBusy && "pulse")} />
          <div>
            <strong>{isBusy ? "Generating" : "Ready"}</strong>
            <small>{displayAvatar}</small>
          </div>
        </div>
      </aside>

      <section className="viewportPanel">
        <div className="viewportToolbar">
          <div>
            <strong>{displayAvatar}</strong>
            <small>{hasAvatar ? `${state?.points?.toLocaleString() ?? "0"} points` : "Waiting for generated avatar"}</small>
          </div>
          <div className="toolbarButtons">
            <button aria-label="Export PLY" disabled={!hasAvatar} onClick={exportPly}>
              <Download size={17} />
            </button>
          </div>
        </div>

        <div className="stage">
          {exportedPly ? (
            <GaussianSplatStage url={exportedPly} />
          ) : hasAvatar ? (
            <img
              className="renderStream"
              src={`${API_BASE}/api/stream.mjpg?width=1280&height=720`}
              alt="FlexAvatar live render"
            />
          ) : (
            <StagePlaceholder selectedPreview={selectedPreview} isBusy={isBusy} />
          )}
          <div className="stageGradient" />
        </div>

        <div className="statusStrip">
          <Metric label="Render FPS" value={state?.renderFps ?? "-"} />
          <Metric label="Animation FPS" value={state?.animationFps ?? "-"} />
          <Metric label="Mode" value={state?.mode ?? "default"} />
          <Metric label="Webcam" value={state?.webcamReady ? "Driving" : "Standby"} />
          {exportedPly && <a href={exportedPly}>Open exported PLY</a>}
        </div>
      </section>

      <aside className="inspector">
        <div className="inspectorHeader">
          <small>Step {currentIndex + 1} of 4</small>
          <h1>{steps[currentIndex].label}</h1>
          <p>{state?.status ?? "Starting FlexAvatar runtime..."}</p>
        </div>

        {activeStep === "input" && (
          <InputStep
            avatars={avatars}
            selectedPreview={selectedPreview}
            selectedInput={selectedInput}
            onFileChange={onFileChange}
            onLoadAvatar={loadAvatar}
          />
        )}
        {activeStep === "generate" && (
          <GenerateStep selectedInput={selectedInput} isBusy={isBusy} onGenerate={generateSelected} />
        )}
        {activeStep === "preview" && (
          <PreviewStep controls={controls} setControls={setControls} hasAvatar={hasAvatar} />
        )}
        {activeStep === "webcam" && (
          <WebcamStep
            webcamOn={webcamOn}
            setWebcamOn={(enabled) => {
              setWebcamOn(enabled);
              setControls((value) => ({ ...value, mode: enabled ? "webcam" : "default" }));
            }}
            videoRef={videoRef}
          />
        )}

        {notice && <div className="notice">{notice}</div>}
      </aside>
    </main>
  );
}

function StagePlaceholder({ selectedPreview, isBusy }) {
  return (
    <div className="stagePlaceholder">
      {selectedPreview ? <img src={selectedPreview} alt="Selected avatar input" /> : <ImagePlus size={42} />}
      <strong>{isBusy ? "Generating avatar..." : "No avatar generated yet"}</strong>
      <span>Choose an input, then wait for generation to finish before the Preview step unlocks.</span>
    </div>
  );
}

function GaussianSplatStage({ url }) {
  const hostRef = React.useRef(null);
  const [loaded, setLoaded] = React.useState(false);

  React.useEffect(() => {
    let viewer;
    let cancelled = false;
    setLoaded(false);

    async function mountViewer() {
      const GaussianSplats3D = await import("@mkkellogg/gaussian-splats-3d");
      if (cancelled || !hostRef.current) return;

      viewer = new GaussianSplats3D.Viewer({
        rootElement: hostRef.current,
        sharedMemoryForWorkers: false,
        cameraUp: [0, 1, 0],
        initialCameraPosition: [0, 0, 1.4],
        initialCameraLookAt: [0, 0, 0],
        selfDrivenMode: true,
        useBuiltInControls: true,
      });
      await viewer.addSplatScene(url, {
        splatAlphaRemovalThreshold: 5,
        showLoadingUI: true,
      });
      if (!cancelled) {
        setLoaded(true);
        viewer.start();
      }
    }

    mountViewer().catch((error) => {
      if (hostRef.current) {
        hostRef.current.dataset.error = error.message;
      }
    });

    return () => {
      cancelled = true;
      if (viewer) {
        viewer.dispose();
      }
    };
  }, [url]);

  return (
    <div className={cx("splatStage", loaded && "loaded")} ref={hostRef}>
      <span>Loading exported Gaussian splat...</span>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <small>{label}</small>
      <strong>{value}</strong>
    </div>
  );
}

function InputStep({ avatars, selectedPreview, selectedInput, onFileChange, onLoadAvatar }) {
  return (
    <div className="stepPane">
      <label className="dropzone">
        <input type="file" accept=".jpg,.jpeg,.png,.mp4" onChange={onFileChange} />
        {selectedPreview ? (
          <img src={selectedPreview} alt="Selected avatar input" />
        ) : (
          <span><Upload size={28} /> Choose portrait image</span>
        )}
      </label>
      <div className="selectedPath">{selectedInput?.avatarName ?? "No image selected"}</div>

      <section className="compactList">
        <header>Existing avatar codes</header>
        {avatars.map((avatar) => (
          <button key={avatar.name} onClick={() => onLoadAvatar(avatar.name)}>
            <span>{avatar.name}</span>
            <ChevronRight size={16} />
          </button>
        ))}
      </section>
    </div>
  );
}

function GenerateStep({ selectedInput, isBusy, onGenerate }) {
  return (
    <div className="stepPane">
      <div className="generationCard">
        <Sparkles size={28} />
        <h2>{selectedInput ? selectedInput.avatarName : "Select an input first"}</h2>
        <p>Pixel3DMM tracking and avatar-code generation run in the Python runtime.</p>
        <button className="primaryButton" disabled={!selectedInput || isBusy} onClick={onGenerate}>
          {isBusy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          Generate avatar
        </button>
      </div>
    </div>
  );
}

function PreviewStep({ controls, setControls, hasAvatar }) {
  const setExpression = (index, value) => {
    setControls((current) => {
      const expression = [...current.expression];
      expression[index] = Number(value);
      return { ...current, mode: "manual", expression };
    });
  };

  return (
    <div className="stepPane">
      {!hasAvatar && (
        <div className="notice">
          Generate or load an avatar first. The viewport stays empty until an avatar exists.
        </div>
      )}
      <div className="segmented">
        <button
          className={controls.mode === "default" ? "selected" : ""}
          onClick={() => setControls((value) => ({ ...value, mode: "default" }))}
        >
          <Video size={16} /> Default video
        </button>
        <button
          className={controls.mode === "manual" ? "selected" : ""}
          onClick={() => setControls((value) => ({ ...value, mode: "manual" }))}
        >
          <SlidersHorizontal size={16} /> Sliders
        </button>
      </div>

      <div className="toggleRow">
        <button
          className={controls.playing ? "iconToggle on" : "iconToggle"}
          onClick={() => setControls((value) => ({ ...value, playing: !value.playing }))}
        >
          {controls.playing ? <Play size={16} /> : <Square size={16} />}
        </button>
        <span>Playback</span>
        <button
          className={controls.lockHead ? "switch on" : "switch"}
          onClick={() => setControls((value) => ({ ...value, lockHead: !value.lockHead }))}
        />
      </div>

      <section className="sliderGroup">
        <header>Expression</header>
        {expressionControls.map((label, index) => (
          <Slider
            key={label}
            label={label}
            value={controls.expression[index]}
            min={-2}
            max={2}
            step={0.01}
            onChange={(value) => setExpression(index, value)}
          />
        ))}
      </section>

      <section className="sliderGroup">
        <header>Camera</header>
        {["yaw", "pitch", "radius"].map((key) => (
          <Slider
            key={key}
            label={key}
            value={controls.camera[key]}
            min={key === "radius" ? 0.55 : -40}
            max={key === "radius" ? 1.8 : 40}
            step={0.01}
            onChange={(value) =>
              setControls((current) => ({
                ...current,
                camera: { ...current.camera, [key]: Number(value) },
              }))
            }
          />
        ))}
      </section>
    </div>
  );
}

function WebcamStep({ webcamOn, setWebcamOn, videoRef }) {
  return (
    <div className="stepPane">
      <div className="webcamFrame">
        <video ref={videoRef} muted playsInline />
        {!webcamOn && <span><Camera size={26} /> Webcam off</span>}
      </div>
      <button className="primaryButton" onClick={() => setWebcamOn(!webcamOn)}>
        {webcamOn ? <Square size={18} /> : <Radio size={18} />}
        {webcamOn ? "Stop webcam drive" : "Drive avatar with webcam"}
      </button>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange }) {
  const handleInput = (event) => onChange(event.target.value);

  return (
    <label className="slider">
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onInput={handleInput} onChange={handleInput} />
      <output>{Number(value).toFixed(2)}</output>
    </label>
  );
}

createRoot(document.getElementById("root")).render(<App />);
