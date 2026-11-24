const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const status = document.getElementById("status");
const panel = document.getElementById("model-panel");
const viewPanel = document.getElementById("view-panel");
const modelCanvas = document.getElementById("model-canvas");
const canvasContext = modelCanvas.getContext("2d", { alpha: false });
let defaultCamera = null;
let camera = null;
let rendering = false;
let needsFrame = false;
let modelGeneration = 0;
let dragPosition = null;

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function renderSummary(summary) {
  document.getElementById("model-title").textContent = summary.model_name;
  const stats = document.getElementById("summary-stats");
  stats.replaceChildren();
  for (const [label, value] of [
    ["File", summary.filename],
    ["Bodies", summary.body_count],
    ["Joints", summary.joint_count],
    ["Chains", summary.chains.length],
    ["Equality constraints", summary.equality_count],
    ["Position coordinates", summary.qpos_size],
  ]) {
    const term = document.createElement("dt");
    term.textContent = label;
    const detail = document.createElement("dd");
    detail.textContent = String(value);
    stats.append(term, detail);
  }

  const bodies = document.getElementById("body-list");
  bodies.replaceChildren();
  for (const body of summary.bodies) {
    const item = document.createElement("li");
    item.textContent = `${body.name} (parent: ${body.parent_id ?? "none"})`;
    bodies.append(item);
  }

  const joints = document.getElementById("joint-list");
  joints.replaceChildren();
  for (const joint of summary.joints) {
    const item = document.createElement("li");
    item.textContent = `${joint.name} · ${joint.type} · body ${joint.body_id}`;
    joints.append(item);
  }

  const chains = document.getElementById("chain-list");
  chains.replaceChildren();
  document.getElementById("chain-empty").hidden = summary.chains.length !== 0;
  for (const chain of summary.chains) {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = `To ${chain.name}: `;
    item.append(title, document.createTextNode(chain.joint_names.join(" → ")));
    const bodyPath = document.createElement("small");
    bodyPath.className = "body-path";
    bodyPath.textContent = `Body path: ${chain.body_names.join(" → ")}`;
    item.append(bodyPath);
    chains.append(item);
  }
  panel.hidden = false;
  defaultCamera = summary.camera;
  camera = { ...defaultCamera, lookat: [...defaultCamera.lookat] };
  modelGeneration += 1;
  viewPanel.hidden = false;
  requestFrame();
}

function requestFrame() {
  needsFrame = true;
  if (!rendering) void renderFrames();
}

async function renderFrames() {
  rendering = true;
  while (needsFrame && camera) {
    needsFrame = false;
    const generation = modelGeneration;
    const params = new URLSearchParams({
      azimuth: String(camera.azimuth),
      elevation: String(camera.elevation),
      distance: String(camera.distance),
    });
    try {
      const response = await fetch(`/api/frame?${params}`);
      if (!response.ok) {
        const result = await response.json();
        throw new Error(result.error?.message || "Could not render this model.");
      }
      const bitmap = await createImageBitmap(await response.blob());
      // Draw in a paint frame so continuous input cannot replace an image
      // before the browser has decoded and displayed it.
      await new Promise((resolve) => requestAnimationFrame(() => {
        if (generation === modelGeneration) {
          canvasContext.drawImage(bitmap, 0, 0, modelCanvas.width, modelCanvas.height);
        }
        bitmap.close();
        resolve();
      }));
    } catch (error) {
      if (generation === modelGeneration) setStatus(error.message, true);
    }
  }
  rendering = false;
}

async function upload(file) {
  if (!file) return;
  setStatus(`Loading ${file.name}…`);
  const form = new FormData();
  form.append("file", file);
  try {
    const response = await fetch("/api/model", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error?.message || "Could not load this MJCF file.");
    }
    renderSummary(result);
    setStatus(`Loaded ${result.filename}.`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

fileInput.addEventListener("change", () => upload(fileInput.files[0]));
modelCanvas.addEventListener("pointerdown", (event) => {
  if (!camera) return;
  dragPosition = { x: event.clientX, y: event.clientY };
  modelCanvas.setPointerCapture(event.pointerId);
});
modelCanvas.addEventListener("pointermove", (event) => {
  if (!dragPosition || !camera) return;
  camera.azimuth += (event.clientX - dragPosition.x) * 0.4;
  camera.elevation = Math.max(-89, Math.min(89, camera.elevation - (event.clientY - dragPosition.y) * 0.4));
  dragPosition = { x: event.clientX, y: event.clientY };
  requestFrame();
});
for (const eventName of ["pointerup", "pointercancel", "lostpointercapture"]) {
  modelCanvas.addEventListener(eventName, () => { dragPosition = null; });
}
modelCanvas.addEventListener("wheel", (event) => {
  if (!camera) return;
  event.preventDefault();
  const minimum = Math.max(0.01, defaultCamera.distance * 0.05);
  const maximum = Math.max(minimum, defaultCamera.distance * 20);
  camera.distance = Math.max(minimum, Math.min(maximum, camera.distance * Math.exp(event.deltaY * 0.001)));
  requestFrame();
}, { passive: false });
document.getElementById("reset-camera").addEventListener("click", () => {
  if (!defaultCamera) return;
  camera = { ...defaultCamera, lookat: [...defaultCamera.lookat] };
  requestFrame();
});
dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragging");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  upload(event.dataTransfer.files[0]);
});
