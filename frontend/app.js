const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const status = document.getElementById("status");
const panel = document.getElementById("model-panel");
const viewPanel = document.getElementById("view-panel");
const modelCanvas = document.getElementById("model-canvas");
const chainSelect = document.getElementById("chain-select");
const poseInput = document.getElementById("pose-input");
const canvasContext = modelCanvas.getContext("2d", { alpha: false });
let defaultCamera = null;
let camera = null;
let rendering = false;
let needsFrame = false;
let modelGeneration = 0;
let dragPosition = null;
let selectedChainId = null;
let poseVersion = 0;
let pendingPose = null;
let sendingPose = false;

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function formatParameter(value) {
  return Math.abs(value) < 1e-10 ? "0" : Number(value).toPrecision(5);
}

function formatError(value) {
  return Number(value).toExponential(2);
}

function renderConstraints(constraints) {
  const list = document.getElementById("constraint-list");
  list.replaceChildren();
  document.getElementById("constraint-panel").hidden = constraints.length === 0;
  for (const constraint of constraints) {
    const item = document.createElement("li");
    const residual = constraint.residual_norm === null
      ? "no active solver residual"
      : `residual ${formatError(constraint.residual_norm)} ${constraint.residual_unit}`;
    const separation = constraint.anchor_separation_m === null
      ? ""
      : ` · anchor gap ${formatError(constraint.anchor_separation_m)} m`;
    item.textContent = `${constraint.name} · ${constraint.type} · ${constraint.active ? "active" : "inactive"} · ${residual}${separation}`;
    list.append(item);
  }
}

function showSelectedChain() {
  for (const item of document.querySelectorAll("#chain-list > li")) {
    item.hidden = item.dataset.chainId !== selectedChainId;
  }
}

function renderSummary(summary) {
  selectedChainId = summary.chains[0]?.id ?? null;
  poseVersion = summary.pose_version;
  pendingPose = null;
  poseInput.value = JSON.stringify(summary.qpos);
  document.getElementById("pose-version").textContent = `version ${poseVersion}`;
  renderConstraints(summary.constraints);
  chainSelect.replaceChildren();
  chainSelect.disabled = summary.chains.length === 0;
  for (const chain of summary.chains) {
    const option = document.createElement("option");
    option.value = chain.id;
    option.textContent = chain.name;
    chainSelect.append(option);
  }
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
    item.dataset.chainId = chain.id;
    const title = document.createElement("strong");
    title.textContent = `To ${chain.name}: `;
    item.append(title, document.createTextNode(chain.joint_names.join(" → ")));
    const bodyPath = document.createElement("small");
    bodyPath.className = "body-path";
    bodyPath.textContent = `Body path: ${chain.body_names.join(" → ")}`;
    item.append(bodyPath);

    const details = document.createElement("details");
    const heading = document.createElement("summary");
    heading.textContent = `Virtual joints (${chain.virtual_joints.length})`;
    const virtualList = document.createElement("ol");
    virtualList.className = "virtual-list";
    for (const row of chain.virtual_joints) {
      const virtualItem = document.createElement("li");
      const qpos = row.qpos_range[1] - row.qpos_range[0] === 1
        ? `qpos[${row.qpos_range[0]}]`
        : `qpos[${row.qpos_range[0]}:${row.qpos_range[1]}]`;
      virtualItem.textContent = `${row.name} · ${row.kind} · ${row.value.toFixed(3)} ${row.unit} · axis (${row.axis.join(", ")}) · ${qpos}`;
      virtualList.append(virtualItem);
    }
    details.append(heading, virtualList);
    for (const warning of chain.virtual_warnings) {
      const note = document.createElement("p");
      note.className = "virtual-warning";
      note.textContent = warning;
      details.append(note);
    }
    item.append(details);

    const assignment = chain.dh_frames;
    const frameDetails = document.createElement("details");
    const frameHeading = document.createElement("summary");
    frameHeading.textContent = `Candidate DH frames (${assignment.frames.length})`;
    const frameNote = document.createElement("p");
    frameNote.className = "body-path";
    frameNote.textContent = "F0 follows the first virtual axis; each later frame follows the next axis. The final frame follows the last joint.";
    const pairList = document.createElement("ul");
    for (const pair of assignment.pairs) {
      const pairItem = document.createElement("li");
      pairItem.textContent = `${pair.from} → ${pair.to}: ${pair.case}, separation ${pair.separation.toPrecision(5)} m`;
      pairList.append(pairItem);
    }
    const frameList = document.createElement("ol");
    frameList.className = "frame-list";
    const vector = (values) => values.map((value) => value.toPrecision(5)).join(", ");
    for (const frame of assignment.frames) {
      const frameItem = document.createElement("li");
      frameItem.textContent = `F${frame.index}: origin (${vector(frame.origin)}), X (${vector(frame.x)}), Y (${vector(frame.y)}), Z (${vector(frame.z)})`;
      frameList.append(frameItem);
    }
    const toolNote = document.createElement("p");
    toolNote.className = "body-path";
    toolNote.textContent = `Fixed tool offset from final frame: (${vector(assignment.tool_transform.slice(0, 3).map((row) => row[3]))}) m`;
    frameDetails.append(frameHeading, frameNote, pairList, frameList, toolNote);
    item.append(frameDetails);

    const table = chain.dh_table;
    const validation = table.validation;
    const tableDetails = document.createElement("details");
    tableDetails.className = "dh-details";
    tableDetails.open = true;
    const tableHeading = document.createElement("summary");
    tableHeading.textContent = `Standard DH table (${table.rows.length} rows) · ${validation.status}`;
    tableHeading.className = validation.status === "verified" ? "validation-ok" : "validation-error";
    const conventionNote = document.createElement("p");
    conventionNote.className = "body-path";
    conventionNote.textContent = "At the reference pose. Revolute rows use θ = θ₀ + (v − v₀); prismatic rows use d = d₀ + (v − v₀), where v is the virtual joint value shown above.";
    const tableWrap = document.createElement("div");
    tableWrap.className = "dh-table-wrap";
    const dhTable = document.createElement("table");
    dhTable.className = "dh-table";
    const thead = document.createElement("thead");
    const header = document.createElement("tr");
    for (const label of ["Joint", "Variable", "θ₀ (rad)", "d₀ (m)", "a (m)", "α (rad)", "θ now (rad)", "d now (m)"]) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      header.append(cell);
    }
    thead.append(header);
    const tbody = document.createElement("tbody");
    for (const row of table.rows) {
      const tr = document.createElement("tr");
      for (const value of [row.name, row.variable === "theta" ? "θ" : "d", row.theta_offset, row.d_offset, row.a, row.alpha, row.theta_offset, row.d_offset]) {
        const cell = document.createElement("td");
        cell.textContent = typeof value === "number" ? formatParameter(value) : value;
        tr.append(cell);
      }
      tbody.append(tr);
    }
    dhTable.append(thead, tbody);
    tableWrap.append(dhTable);
    const validationNote = document.createElement("p");
    validationNote.className = validation.status === "verified" ? "validation-ok" : "validation-error";
    validationNote.textContent = `${validation.status === "verified" ? "Verified" : "Validation failed"} against MuJoCo tree FK at ${validation.sample_count} poses · max position error ${formatError(validation.max_fk_position_error_m)} m · max orientation error ${formatError(validation.max_fk_orientation_error_rad)} rad.`;
    const constraintNote = document.createElement("p");
    constraintNote.className = "body-path";
    constraintNote.textContent = "Equality constraints are outside this open-chain DH comparison.";
    tableDetails.append(tableHeading, conventionNote, tableWrap, validationNote);
    const currentError = document.createElement("p");
    currentError.className = "pose-fk-error body-path";
    tableDetails.append(currentError);
    if (summary.equality_count > 0) tableDetails.append(constraintNote);
    item.append(tableDetails);
    chains.append(item);
  }
  showSelectedChain();
  panel.hidden = false;
  defaultCamera = summary.camera;
  camera = { ...defaultCamera, lookat: [...defaultCamera.lookat] };
  modelGeneration += 1;
  viewPanel.hidden = false;
  requestFrame();
}

function applyPoseSnapshot(snapshot) {
  if (snapshot.pose_version < poseVersion || snapshot.chain_id !== selectedChainId) return;
  poseVersion = snapshot.pose_version;
  poseInput.value = JSON.stringify(snapshot.qpos);
  document.getElementById("pose-version").textContent = `version ${poseVersion}`;
  renderConstraints(snapshot.constraints);
  if (!selectedChainId) return;
  const item = [...document.querySelectorAll("#chain-list > li")]
    .find((entry) => entry.dataset.chainId === selectedChainId);
  if (!item) return;
  const virtualList = item.querySelector(".virtual-list");
  for (const [index, row] of snapshot.virtual_joints.entries()) {
    const qpos = row.qpos_range[1] - row.qpos_range[0] === 1
      ? `qpos[${row.qpos_range[0]}]`
      : `qpos[${row.qpos_range[0]}:${row.qpos_range[1]}]`;
    virtualList.children[index].textContent = `${row.name} · ${row.kind} · ${row.value.toFixed(3)} ${row.unit} · axis (${row.axis.join(", ")}) · ${qpos}`;
  }
  const virtualDetails = virtualList.parentElement;
  for (const warning of virtualDetails.querySelectorAll(".virtual-warning")) warning.remove();
  for (const warning of snapshot.virtual_warnings) {
    const note = document.createElement("p");
    note.className = "virtual-warning";
    note.textContent = warning;
    virtualDetails.append(note);
  }
  const vector = (values) => values.map((value) => value.toPrecision(5)).join(", ");
  const frameList = item.querySelector(".frame-list");
  for (const [index, frame] of snapshot.frames.entries()) {
    frameList.children[index].textContent = `F${frame.index}: origin (${vector(frame.origin)}), X (${vector(frame.x)}), Y (${vector(frame.y)}), Z (${vector(frame.z)})`;
  }
  const rows = item.querySelectorAll(".dh-table tbody tr");
  for (const [index, values] of snapshot.dh_values.entries()) {
    rows[index].cells[6].textContent = formatParameter(values.theta);
    rows[index].cells[7].textContent = formatParameter(values.d);
  }
  item.querySelector(".pose-fk-error").textContent = `Current-pose FK error: ${formatError(snapshot.fk_position_error_m)} m, ${formatError(snapshot.fk_orientation_error_rad)} rad.`;
}

async function refreshSelectedChain() {
  const generation = modelGeneration;
  const chainId = selectedChainId;
  try {
    const params = chainId ? `?chain_id=${encodeURIComponent(chainId)}` : "";
    const response = await fetch(`/api/pose${params}`);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error?.message || "Could not read the pose.");
    if (generation !== modelGeneration || chainId !== selectedChainId) return;
    applyPoseSnapshot(result);
    requestFrame();
  } catch (error) {
    if (generation === modelGeneration) setStatus(error.message, true);
  }
}

async function sendPendingPoses() {
  if (sendingPose) return;
  sendingPose = true;
  while (pendingPose) {
    const request = pendingPose;
    pendingPose = null;
    try {
      const response = await fetch("/api/pose", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ qpos: request.qpos, chain_id: request.chainId }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error?.message || "Could not apply the pose.");
      if (request.generation !== modelGeneration) continue;
      applyPoseSnapshot(result);
      requestFrame();
      setStatus(`Pose ${result.pose_version} applied.`);
    } catch (error) {
      if (request.generation === modelGeneration) setStatus(error.message, true);
    }
  }
  sendingPose = false;
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
    const chainId = selectedChainId;
    const params = new URLSearchParams({
      azimuth: String(camera.azimuth),
      elevation: String(camera.elevation),
      distance: String(camera.distance),
    });
    if (chainId) params.set("chain_id", chainId);
    try {
      const response = await fetch(`/api/frame?${params}`);
      if (!response.ok) {
        const result = await response.json();
        throw new Error(result.error?.message || "Could not render this model.");
      }
      const framePoseVersion = Number(response.headers.get("X-Pose-Version"));
      const bitmap = await createImageBitmap(await response.blob());
      // Draw in a paint frame so continuous input cannot replace an image
      // before the browser has decoded and displayed it.
      await new Promise((resolve) => requestAnimationFrame(() => {
        if (generation === modelGeneration && chainId === selectedChainId && framePoseVersion >= poseVersion) {
          canvasContext.drawImage(bitmap, 0, 0, modelCanvas.width, modelCanvas.height);
        } else if (generation === modelGeneration) {
          needsFrame = true;
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
chainSelect.addEventListener("change", () => {
  selectedChainId = chainSelect.value || null;
  showSelectedChain();
  void refreshSelectedChain();
});
document.getElementById("apply-pose").addEventListener("click", () => {
  try {
    const qpos = JSON.parse(poseInput.value);
    pendingPose = { qpos, chainId: selectedChainId, generation: modelGeneration };
    void sendPendingPoses();
  } catch (error) {
    setStatus(`Invalid JSON: ${error.message}`, true);
  }
});
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
