const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const status = document.getElementById("status");
const panel = document.getElementById("model-panel");

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
  panel.hidden = false;
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
