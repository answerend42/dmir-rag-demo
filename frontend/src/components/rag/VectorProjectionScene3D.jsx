/**
 * @file VectorProjectionScene3D.jsx
 * @brief 使用 Three.js 渲染后端返回的三维向量投影坐标。
 */
/* eslint-disable react/prop-types */
import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const SCENE_SCALE = 8;
const POINT_COLORS = [
  0x0f766e,
  0xb45309,
  0x2563eb,
  0xdc2626,
  0x7c3aed,
  0x16a34a,
];

const clampUnit = (value) => {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return 0.5;
  }
  return Math.min(Math.max(numericValue, 0), 1);
};

const pointPosition = (point) => new THREE.Vector3(
  (clampUnit(point.x) - 0.5) * SCENE_SCALE,
  (clampUnit(point.y) - 0.5) * SCENE_SCALE,
  (clampUnit(point.z) - 0.5) * SCENE_SCALE,
);

const colorForPage = (pageNumber) => {
  const pageIndex = Math.max(Number(pageNumber) || 1, 1) - 1;
  return POINT_COLORS[pageIndex % POINT_COLORS.length];
};

const makeTextSprite = (text, color = '#0f172a', scale = 0.7) => {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 96;
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.font = '700 40px system-ui, -apple-system, BlinkMacSystemFont, sans-serif';
  context.fillStyle = color;
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(scale * 2.2, scale * 0.8, 1);
  return sprite;
};

const makeAxisLine = (from, to, color) => {
  const geometry = new THREE.BufferGeometry().setFromPoints([from, to]);
  const material = new THREE.LineBasicMaterial({ color });
  return new THREE.Line(geometry, material);
};

/**
 * @brief 渲染可旋转的三维向量投影场景。
 * @param {object} props 组件属性。
 * @returns {JSX.Element} 三维投影视图容器。
 */
const VectorProjectionScene3D = ({
  points,
  queryPoint,
  hitMap,
  zoom,
  selectedIndex,
  hoveredIndex,
  onSelectIndex,
  onHoverIndex,
  getPointChunkId,
  getPointLabel,
}) => {
  const containerRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return undefined;
    }
    container.innerHTML = '';

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fafc);

    const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 100);
    camera.position.set(7, 6, 9);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.domElement.className = 'block h-full w-full';
    renderer.domElement.setAttribute('aria-label', '嵌入向量三维投影场景');
    renderer.domElement.setAttribute('data-vector-projection-canvas', '3d');
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.55;
    controls.zoomSpeed = 0.7;
    controls.target.set(0, 0, 0);

    scene.add(new THREE.HemisphereLight(0xffffff, 0xcbd5e1, 2.2));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.8);
    directionalLight.position.set(4, 7, 6);
    scene.add(directionalLight);

    const rootGroup = new THREE.Group();
    rootGroup.scale.setScalar(Number.isFinite(Number(zoom)) ? Number(zoom) : 1);
    scene.add(rootGroup);

    const grid = new THREE.GridHelper(9, 9, 0xcbd5e1, 0xe2e8f0);
    grid.position.y = -4.5;
    rootGroup.add(grid);
    rootGroup.add(makeAxisLine(new THREE.Vector3(-4.7, 0, 0), new THREE.Vector3(4.7, 0, 0), 0x64748b));
    rootGroup.add(makeAxisLine(new THREE.Vector3(0, -4.7, 0), new THREE.Vector3(0, 4.7, 0), 0x64748b));
    rootGroup.add(makeAxisLine(new THREE.Vector3(0, 0, -4.7), new THREE.Vector3(0, 0, 4.7), 0x64748b));

    [
      ['x', new THREE.Vector3(5.0, 0, 0)],
      ['y', new THREE.Vector3(0, 5.0, 0)],
      ['z', new THREE.Vector3(0, 0, 5.0)],
    ].forEach(([label, position]) => {
      const sprite = makeTextSprite(label, '#475569', 0.45);
      sprite.position.copy(position);
      rootGroup.add(sprite);
    });

    const pointMeshes = [];
    const sphereGeometry = new THREE.SphereGeometry(0.105, 18, 18);
    const hitGeometry = new THREE.SphereGeometry(0.16, 22, 22);
    const selectedGeometry = new THREE.SphereGeometry(0.19, 24, 24);
    const queryGeometry = new THREE.OctahedronGeometry(0.28, 0);

    points.forEach((point) => {
      const metadata = point.embedding?.metadata || {};
      const hitInfo = hitMap.get(getPointChunkId(point));
      const isSelected = selectedIndex === point.index;
      const isHovered = hoveredIndex === point.index;
      const isHit = Boolean(hitInfo);
      const muted = hitMap.size > 0 && !isHit;
      const geometry = isSelected || isHovered ? selectedGeometry : isHit ? hitGeometry : sphereGeometry;
      const material = new THREE.MeshStandardMaterial({
        color: isHit ? 0x0f766e : colorForPage(metadata.page_number || metadata.page),
        roughness: 0.46,
        metalness: 0.04,
        transparent: true,
        opacity: muted ? 0.28 : 0.92,
        emissive: isSelected || isHovered || isHit ? 0x052e2b : 0x000000,
        emissiveIntensity: isSelected || isHovered ? 0.32 : isHit ? 0.18 : 0,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(pointPosition(point));
      mesh.userData = {
        index: point.index,
        label: hitInfo ? `${getPointLabel(point)} · Top ${hitInfo.rank}` : getPointLabel(point),
      };
      rootGroup.add(mesh);
      pointMeshes.push(mesh);

      if (hitInfo) {
        const label = makeTextSprite(`#${hitInfo.rank}`, '#047857', 0.48);
        label.position.copy(mesh.position).add(new THREE.Vector3(0.26, 0.28, 0.16));
        rootGroup.add(label);
      }
    });

    if (queryPoint) {
      const queryMaterial = new THREE.MeshStandardMaterial({
        color: 0x0f172a,
        roughness: 0.38,
        metalness: 0.08,
        emissive: 0x0f172a,
        emissiveIntensity: 0.28,
      });
      const queryMesh = new THREE.Mesh(queryGeometry, queryMaterial);
      queryMesh.position.copy(pointPosition(queryPoint));
      queryMesh.userData = { index: null, label: queryPoint.label || '用户查询向量' };
      rootGroup.add(queryMesh);

      const queryLabel = makeTextSprite('Query', '#0f172a', 0.55);
      queryLabel.position.copy(queryMesh.position).add(new THREE.Vector3(0.45, 0.45, 0.2));
      rootGroup.add(queryLabel);
    }

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const updatePointer = (event, shouldSelect = false) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const intersections = raycaster.intersectObjects(pointMeshes, false);
      const nextIndex = intersections[0]?.object?.userData?.index;
      onHoverIndex(Number.isInteger(nextIndex) ? nextIndex : null);
      if (shouldSelect && Number.isInteger(nextIndex)) {
        onSelectIndex(nextIndex);
      }
    };

    const handlePointerLeave = () => onHoverIndex(null);
    renderer.domElement.addEventListener('pointermove', updatePointer);
    renderer.domElement.addEventListener('pointerleave', handlePointerLeave);
    renderer.domElement.addEventListener('click', (event) => updatePointer(event, true));

    const resize = () => {
      const width = container.clientWidth || 720;
      const height = container.clientHeight || 460;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    const resizeObserver = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(resize) : null;
    resizeObserver?.observe(container);
    window.addEventListener('resize', resize);
    resize();

    let frameId = 0;
    const animate = () => {
      frameId = window.requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      window.cancelAnimationFrame(frameId);
      window.removeEventListener('resize', resize);
      resizeObserver?.disconnect();
      renderer.domElement.removeEventListener('pointermove', updatePointer);
      renderer.domElement.removeEventListener('pointerleave', handlePointerLeave);
      renderer.dispose();
      controls.dispose();
      scene.traverse((object) => {
        if (object.geometry) {
          object.geometry.dispose();
        }
        if (object.material) {
          if (object.material.map) {
            object.material.map.dispose();
          }
          object.material.dispose();
        }
      });
      container.innerHTML = '';
    };
  }, [
    points,
    queryPoint,
    hitMap,
    zoom,
    selectedIndex,
    hoveredIndex,
    onSelectIndex,
    onHoverIndex,
    getPointChunkId,
    getPointLabel,
  ]);

  return (
    <div
      ref={containerRef}
      className="h-full min-h-[420px] w-full bg-slate-50"
    />
  );
};

export default VectorProjectionScene3D;
