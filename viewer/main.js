import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";

const viewerShell = document.getElementById("viewerShell");
const leftPaneEl = document.getElementById("leftPane");
const rightPaneEl = document.getElementById("rightPane");
const leftInput = document.getElementById("leftPanoramaInput");
const rightInput = document.getElementById("rightPanoramaInput");
const splitToggle = document.getElementById("splitToggle");
const emptyState = document.getElementById("emptyState");

const loader = new THREE.TextureLoader();
const sphereGeometry = new THREE.SphereGeometry(500, 60, 40);
sphereGeometry.scale(-1, 1, 1); // flip normals so texture is on the inside

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const setEmptyStateVisibility = (visible) => {
  if (!emptyState) return;
  emptyState.hidden = !visible;
};

// Shared camera state so angles stay aligned across panes.
let lon = 0;
let lat = 0;
let fov = 75;

class PanoramaPane {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(
      fov,
      (container.clientWidth || window.innerWidth) /
        (container.clientHeight || window.innerHeight),
      0.1,
      2000
    );
    this.camera.position.set(0, 0, 0.01);

    this.material = new THREE.MeshBasicMaterial();
    this.mesh = new THREE.Mesh(sphereGeometry, this.material);
    this.scene.add(this.mesh);

    this.objectUrlInUse = null;
    this.resize();
  }

  async setTexture(url, { isObjectUrl = false } = {}) {
    const nextTexture = await loader.loadAsync(url);
    nextTexture.colorSpace = THREE.SRGBColorSpace;

    const oldTexture = this.material.map;
    this.material.map = nextTexture;
    this.material.needsUpdate = true;

    oldTexture?.dispose();
    if (this.objectUrlInUse && this.objectUrlInUse !== url) {
      URL.revokeObjectURL(this.objectUrlInUse);
    }
    this.objectUrlInUse = isObjectUrl ? url : null;
  }

  resize() {
    const { clientWidth, clientHeight } = this.container;
    this.renderer.setSize(clientWidth, clientHeight);
    this.camera.aspect = clientWidth / Math.max(clientHeight, 1);
    this.camera.updateProjectionMatrix();
  }

  isVisible() {
    return (
      this.container.offsetParent !== null &&
      this.container.clientWidth > 0 &&
      this.container.clientHeight > 0
    );
  }

  render(target, currentFov) {
    if (!this.isVisible()) return;
    this.camera.fov = currentFov;
    this.camera.lookAt(target);
    this.camera.updateProjectionMatrix();
    this.renderer.render(this.scene, this.camera);
  }
}

const leftPane = new PanoramaPane(leftPaneEl);
const rightPane = new PanoramaPane(rightPaneEl);
const panes = [leftPane, rightPane];

async function loadDefaultPanorama() {
  try {
    await Promise.all(panes.map((pane) => pane.setTexture("./pano.jpg")));
    setEmptyStateVisibility(false);
    return true;
  } catch (err) {
    console.warn("Default panorama not found. Load one to begin.", err);
    setEmptyStateVisibility(true);
    return false;
  }
}
await loadDefaultPanorama();

function attachFileInput(input, pane) {
  input?.addEventListener("change", async (event) => {
    const [file] = event.target.files;
    if (!file) return;

    const objectUrl = URL.createObjectURL(file);
    try {
      await pane.setTexture(objectUrl, { isObjectUrl: true });
      setEmptyStateVisibility(false);
    } catch (err) {
      console.error("Could not load selected panorama:", err);
      URL.revokeObjectURL(objectUrl);
    } finally {
      event.target.value = "";
    }
  });
}

attachFileInput(leftInput, leftPane);
attachFileInput(rightInput, rightPane);

let splitActive = false;
function applySplitMode(nextState) {
  splitActive = nextState;
  viewerShell.classList.toggle("is-split", splitActive);
  if (splitToggle) {
    splitToggle.textContent = splitActive ? "Split view: On" : "Split view: Off";
  }
  resizeAll();
}

splitToggle?.addEventListener("click", () => applySplitMode(!splitActive));
applySplitMode(false);

function resizeAll() {
  panes.forEach((pane) => pane.resize());
}

window.addEventListener("resize", resizeAll);

// Simple orbit controls shared across panes
let isDragging = false;
let prevX = 0;
let prevY = 0;
let activePointerId = null;
let activeSurface = null;

function onPointerDown(e) {
  isDragging = true;
  activePointerId = e.pointerId;
  activeSurface = e.currentTarget;
  prevX = e.clientX;
  prevY = e.clientY;
  activeSurface?.setPointerCapture?.(e.pointerId);
}

function onPointerMove(e) {
  if (!isDragging || e.pointerId !== activePointerId) return;
  const dx = e.clientX - prevX;
  const dy = e.clientY - prevY;
  prevX = e.clientX;
  prevY = e.clientY;

  lon -= dx * 0.1;
  lat += dy * 0.1;
  lat = clamp(lat, -85, 85);
}

function onPointerUp(e) {
  if (e.pointerId !== activePointerId) return;
  isDragging = false;
  activeSurface?.releasePointerCapture?.(e.pointerId);
  activePointerId = null;
  activeSurface = null;
}

function onWheel(e) {
  fov = clamp(fov + e.deltaY * 0.05, 30, 100);
}

panes.forEach((pane) => {
  pane.renderer.domElement.addEventListener("pointerdown", onPointerDown);
  pane.renderer.domElement.addEventListener("pointermove", onPointerMove);
});
window.addEventListener("pointerup", onPointerUp, { passive: true });
window.addEventListener("wheel", onWheel, { passive: true });

// Touch pinch zoom (basic)
let lastDist = null;
function dist(t0, t1) {
  const dx = t0.clientX - t1.clientX;
  const dy = t0.clientY - t1.clientY;
  return Math.hypot(dx, dy);
}
window.addEventListener(
  "touchmove",
  (e) => {
    if (e.touches.length === 2) {
      const d = dist(e.touches[0], e.touches[1]);
      if (lastDist != null) {
        const delta = lastDist - d;
        fov = clamp(fov + delta * 0.1, 30, 100);
      }
      lastDist = d;
    }
  },
  { passive: true }
);
window.addEventListener(
  "touchend",
  () => {
    lastDist = null;
  },
  { passive: true }
);

const target = new THREE.Vector3();
function animate() {
  requestAnimationFrame(animate);

  const phi = THREE.MathUtils.degToRad(90 - lat);
  const theta = THREE.MathUtils.degToRad(lon);

  target.set(
    Math.sin(phi) * Math.cos(theta),
    Math.cos(phi),
    Math.sin(phi) * Math.sin(theta)
  );

  panes.forEach((pane) => pane.render(target, fov));
}
animate();
