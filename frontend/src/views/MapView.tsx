import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useSimulation, type SimStep } from "../simulation/SimulationContext";

const BASIN_POLYGON = {
  type: "Feature",
  geometry: {
    type: "Polygon",
    coordinates: [[[86.65, 27.65], [87.0, 27.65], [87.0, 27.98], [86.65, 27.98], [86.65, 27.65]]],
  },
  properties: { name: "Dudh Koshi / Imja" },
};

const ASSET_MARKERS = [
  { id: "village-2", type: "village", name: "Chhukung", lon: 86.86, lat: 27.90, status: "amber", pop: 1240, distance: 210, buffer: 100 },
  { id: "BR-12", type: "bridge", name: "Hillary Bridge", lon: 86.85, lat: 27.91, status: "amber", pop: 0, distance: 60, buffer: 75 },
  { id: "RD-4", type: "road", name: "Road 4", lon: 86.86, lat: 27.90, status: "amber", pop: 0, distance: 40, buffer: 50 },
  { id: "well-3", type: "well", name: "Well 3", lon: 86.86, lat: 27.89, status: "red", pop: 0, distance: 90, buffer: 100 },
];

const STATUS_COLOR: Record<string, string> = { green: "#22C55E", amber: "#F59E0B", red: "#EF4444" };

const STEP_LABELS: Record<SimStep, string> = {
  before: "Baseline — 2025-11-22",
  "obs-1": "Obs 1 — 2026-08-23",
  "obs-2": "Obs 2 — 2026-08-29",
  "obs-3": "Obs 3 — 2026-09-04",
};

interface MapViewProps {
  onJumpToReview?: () => void;
}

export default function MapView({ onJumpToReview }: MapViewProps = {}) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markersRef = useRef<maplibregl.Marker[]>([]);
  const [swipePct, setSwipePct] = useState(50);
  const sim = useSimulation();

  const [layers, setLayers] = useState({
    basin: true, dem: false, optical: true, sar: false,
    water: true, corridor: true, assets: true,
  });

  // SAR layer reveals after router fires (step >= obs-2)
  const sarRevealed = sim.step === "obs-2" || sim.step === "obs-3";

  // Find selected asset from context
  const selectedAsset = ASSET_MARKERS.find((a) => a.id === sim.selectedAssetId) ?? null;

  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8, sources: {},
        layers: [{ id: "background", type: "background", paint: { "background-color": "#0a0f1e" } }],
      },
      center: [86.82, 27.88], zoom: 11,
    });
    mapRef.current = map;

    map.on("load", () => {
      map.addSource("basin", { type: "geojson", data: BASIN_POLYGON as any });
      map.addLayer({ id: "basin-fill", type: "fill", source: "basin", paint: { "fill-color": "#06B6D4", "fill-opacity": 0.08 }, layout: { visibility: layers.basin ? "visible" : "none" } });
      map.addLayer({ id: "basin-border", type: "line", source: "basin", paint: { "line-color": "#06B6D4", "line-width": 2 }, layout: { visibility: layers.basin ? "visible" : "none" } });

      map.addSource("water", {
        type: "geojson",
        data: { type: "Feature", geometry: { type: "Polygon", coordinates: [[[86.80, 27.87], [86.84, 27.87], [86.84, 27.90], [86.80, 27.90], [86.80, 27.87]]] }, properties: {} } as any,
      });
      map.addLayer({ id: "water-fill", type: "fill", source: "water", paint: { "fill-color": "#3B82F6", "fill-opacity": 0.4 }, layout: { visibility: layers.water ? "visible" : "none" } });

      // D8 corridor (mock line)
      map.addSource("corridor", {
        type: "geojson",
        data: { type: "Feature", geometry: { type: "LineString", coordinates: [[86.82, 27.88], [86.85, 27.91]] }, properties: {} } as any,
      });
      map.addLayer({ id: "corridor-line", type: "line", source: "corridor", paint: { "line-color": "#F59E0B", "line-width": 3, "line-dasharray": [2, 1] }, layout: { visibility: layers.corridor ? "visible" : "none" } });

      // Asset markers
      ASSET_MARKERS.forEach((asset) => {
        const el = document.createElement("div");
        el.style.cssText = `width: 14px; height: 14px; border-radius: 50%; background: ${STATUS_COLOR[asset.status]}; border: 2px solid #0F172A; cursor: pointer;`;
        el.title = asset.name;
        el.onclick = () => sim.selectAsset(asset.id);
        const marker = new maplibregl.Marker(el).setLngLat([asset.lon, asset.lat]).addTo(map);
        markersRef.current.push(marker);
      });
    });

    return () => { map.remove(); mapRef.current = null; markersRef.current = []; };
  }, []);

  // flyTo when selectedAssetId changes
  useEffect(() => {
    if (selectedAsset && mapRef.current) {
      mapRef.current.flyTo({ center: [selectedAsset.lon, selectedAsset.lat], zoom: 13, duration: 1000 });
    }
  }, [sim.selectedAssetId]);

  const toggleLayer = (key: keyof typeof layers) => {
    setLayers((prev) => {
      const next = { ...prev, [key]: !prev[key] };
      const map = mapRef.current;
      if (map) {
        if (key === "basin") { map.setLayoutProperty("basin-fill", "visibility", next.basin ? "visible" : "none"); map.setLayoutProperty("basin-border", "visibility", next.basin ? "visible" : "none"); }
        if (key === "water") map.setLayoutProperty("water-fill", "visibility", next.water ? "visible" : "none");
        if (key === "corridor") map.setLayoutProperty("corridor-line", "visibility", next.corridor ? "visible" : "none");
      }
      return next;
    });
  };

  const handleSwipe = (e: React.MouseEvent | React.TouchEvent) => {
    const container = e.currentTarget as HTMLElement;
    const rect = container.getBoundingClientRect();
    const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
    setSwipePct(Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100)));
  };

  return (
    <div style={{ display: "flex", gap: 12, height: "100%" }}>
      {/* Left dock — 7 layer toggles */}
      <div className="card" style={{ width: 220, flexShrink: 0 }}>
        <div className="card-title">Layers</div>
        {([
          ["basin", "Basin AOI"], ["dem", "DEM hillshade"], ["optical", "Optical baseline"],
          ["sar", "SAR backscatter"], ["water", "Water expansion"], ["corridor", "D8 + OSM corridor"],
          ["assets", "OSM assets"],
        ] as [keyof typeof layers, string][]).map(([key, label]) => (
          <label key={key} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, cursor: "pointer", fontSize: 13 }}>
            <input
              type="checkbox"
              checked={key === "sar" ? (layers.sar && sarRevealed) : layers[key]}
              onChange={() => toggleLayer(key)}
              disabled={key === "sar" && !sarRevealed}
            />
            {label}
            {key === "sar" && sarRevealed && <span style={{ color: "var(--accent)", fontSize: 10 }}>⚡ revealed</span>}
            {key === "sar" && !sarRevealed && <span style={{ color: "var(--text-dim)", fontSize: 10 }}>(locked)</span>}
          </label>
        ))}
        <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--panel-2)" }}>
          <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 4 }}>Opacity fallback</div>
          <input type="range" min="0" max="100" defaultValue="100" style={{ width: "100%" }} />
        </div>
      </div>

      {/* Map canvas + step badge */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 12 }}>
        <div style={{ flex: 1, position: "relative", borderRadius: 8, overflow: "hidden" }}>
          <div ref={mapContainer} className="map-container" />
          {/* Step badge overlay */}
          <div style={{ position: "absolute", top: 12, left: 12, zIndex: 1, padding: "6px 12px", borderRadius: 4, background: "rgba(15,23,42,0.85)", border: "1px solid var(--panel-2)", fontSize: 12, color: "var(--accent)", fontWeight: 600 }}>
            {STEP_LABELS[sim.step]}
          </div>
        </div>

        {/* Swipe compare */}
        <div className="card" style={{ flexShrink: 0 }}>
          <div className="card-title">Before / After Swipe — water expansion</div>
          <div className="swipe-container" onMouseMove={handleSwipe} onTouchMove={handleSwipe}>
            <div className="swipe-layer" style={{ background: "linear-gradient(135deg, #1a2a4a, #0a0f1e)" }}>
              <div style={{ position: "absolute", top: 8, left: 8, fontSize: 11, color: "#94A3B8" }}>BEFORE (baseline)</div>
            </div>
            <div className="swipe-layer" style={{ background: "linear-gradient(135deg, #1a3a5a, #0a1f3e)", clipPath: `inset(0 0 0 ${swipePct}%)` }}>
              <div style={{ position: "absolute", top: 8, right: 8, fontSize: 11, color: "#3B82F6" }}>AFTER</div>
            </div>
            <div className="swipe-handle" style={{ left: `${swipePct}%` }} />
          </div>
        </div>
      </div>

      {/* Right dock — legend + asset detail */}
      <div style={{ width: 280, flexShrink: 0, display: "flex", flexDirection: "column", gap: 12 }}>
        <div className="card">
          <div className="card-title">Asset Legend</div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6, fontSize: 13 }}><span style={{ width: 10, height: 10, borderRadius: "50%", background: "#22C55E" }} /> Safe</div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6, fontSize: 13 }}><span style={{ width: 10, height: 10, borderRadius: "50%", background: "#F59E0B" }} /> Buffered (within 100m)</div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13 }}><span style={{ width: 10, height: 10, borderRadius: "50%", background: "#EF4444" }} /> Inundated</div>
        </div>

        {selectedAsset && (
          <div className="card">
            <div className="card-title">{selectedAsset.name}</div>
            <div style={{ fontSize: 13, color: "var(--text-dim)", marginBottom: 4 }}>Type: {selectedAsset.type}</div>
            <div style={{ fontSize: 13, color: "var(--text-dim)", marginBottom: 4 }}>Buffer: ±{selectedAsset.buffer} m</div>
            <div style={{ fontSize: 13, color: "var(--text-dim)", marginBottom: 4 }}>{selectedAsset.distance} m from corridor</div>
            {selectedAsset.pop > 0 && <div style={{ fontSize: 13, color: "var(--text-dim)", marginBottom: 4 }}>Population: {selectedAsset.pop}</div>}
            <div style={{ fontSize: 13, marginBottom: 12 }}>
              Status: <span className={`badge ${selectedAsset.status === "green" ? "badge-safe" : selectedAsset.status === "amber" ? "badge-warn" : "badge-danger"}`}>
                {selectedAsset.status === "green" ? "SAFE" : selectedAsset.status === "amber" ? "BUFFERED" : "INUNDATED"}
              </span>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn btn-ghost" style={{ fontSize: 12, padding: "4px 10px" }} onClick={() => mapRef.current?.flyTo({ center: [selectedAsset.lon, selectedAsset.lat], zoom: 14, duration: 1000 })}>
                Fly to
              </button>
              {onJumpToReview && (
                <button className="btn btn-ghost" style={{ fontSize: 12, padding: "4px 10px" }} onClick={onJumpToReview}>
                  Review →
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
