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
const STATUS_LABEL: Record<string, string> = { green: "Safe", amber: "Buffered", red: "Inundated" };
const STATUS_BADGE: Record<string, string> = { green: "badge-safe", amber: "badge-warn", red: "badge-danger" };

const STEP_LABELS: Record<SimStep, string> = {
  before: "Baseline — Nov 22, 2025",
  "obs-1": "Obs 1 — Aug 23, 2026",
  "obs-2": "Obs 2 — Aug 29, 2026",
  "obs-3": "Obs 3 — Sep 4, 2026",
};

const LAYER_LABELS: { key: keyof MapViewLayers; label: string }[] = [
  { key: "basin", label: "Basin boundary" },
  { key: "optical", label: "Optical satellite" },
  { key: "sar", label: "Radar (SAR)" },
  { key: "water", label: "Water expansion" },
  { key: "corridor", label: "Flow path" },
  { key: "assets", label: "Infrastructure" },
];

type MapViewLayers = {
  basin: boolean; optical: boolean; sar: boolean;
  water: boolean; corridor: boolean; assets: boolean;
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

  const [layers, setLayers] = useState<MapViewLayers>({
    basin: true, optical: true, sar: false,
    water: true, corridor: true, assets: true,
  });

  const sarRevealed = sim.step === "obs-2" || sim.step === "obs-3";
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

      map.addSource("corridor", {
        type: "geojson",
        data: { type: "Feature", geometry: { type: "LineString", coordinates: [[86.82, 27.88], [86.85, 27.91]] }, properties: {} } as any,
      });
      map.addLayer({ id: "corridor-line", type: "line", source: "corridor", paint: { "line-color": "#F59E0B", "line-width": 3, "line-dasharray": [2, 1] }, layout: { visibility: layers.corridor ? "visible" : "none" } });

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

  useEffect(() => {
    if (selectedAsset && mapRef.current) {
      mapRef.current.flyTo({ center: [selectedAsset.lon, selectedAsset.lat], zoom: 13, duration: 1000 });
    }
  }, [sim.selectedAssetId]);

  const toggleLayer = (key: keyof MapViewLayers) => {
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
      {/* Left dock — layer toggles */}
      <div className="card" style={{ width: 200, flexShrink: 0, display: "flex", flexDirection: "column" }}>
        <div className="card-title">Layers</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {LAYER_LABELS.map(({ key, label }) => {
            const isLocked = key === "sar" && !sarRevealed;
            const checked = key === "sar" ? (layers.sar && sarRevealed) : layers[key];
            return (
              <label
                key={key}
                style={{
                  display: "flex", alignItems: "center", gap: 10,
                  cursor: isLocked ? "not-allowed" : "pointer",
                  fontSize: 14, opacity: isLocked ? 0.5 : 1,
                }}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => !isLocked && toggleLayer(key)}
                  disabled={isLocked}
                  style={{ width: 16, height: 16, accentColor: "var(--accent)" }}
                />
                <span>{label}</span>
                {isLocked && <span style={{ color: "var(--text-dim)", fontSize: 12 }}>(locked)</span>}
              </label>
            );
          })}
        </div>
        <div style={{ marginTop: 16, paddingTop: 16, borderTop: "1px solid var(--panel-2)" }}>
          <div style={{ fontSize: 13, color: "var(--text-dim)", marginBottom: 6 }}>Opacity</div>
          <input type="range" min="0" max="100" defaultValue="100" style={{ width: "100%", accentColor: "var(--accent)" }} />
        </div>
      </div>

      {/* Map canvas + swipe compare */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
        <div style={{ flex: 1, position: "relative", borderRadius: 8, overflow: "hidden" }}>
          <div ref={mapContainer} className="map-container" />
          <div className="map-step-label">{STEP_LABELS[sim.step]}</div>
        </div>

        <div className="card" style={{ flexShrink: 0 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
            <span className="card-title" style={{ margin: 0 }}>Before / After</span>
            <span style={{ fontSize: 12, color: "var(--text-dim)" }}>Drag to compare</span>
          </div>
          <div className="swipe-container" onMouseMove={handleSwipe} onTouchMove={handleSwipe}>
            <div className="swipe-before" style={{ width: `${swipePct}%` }}>
              <span style={{ fontSize: 13, color: "var(--text-dim)" }}>Before — 3.0 km²</span>
            </div>
            <div className="swipe-after">
              <span style={{ fontSize: 13, color: "var(--accent)" }}>After — 4.1 km² (+14.3%)</span>
            </div>
            <div className="swipe-handle" style={{ left: `${swipePct}%` }} />
          </div>
        </div>
      </div>

      {/* Right dock — legend + asset detail */}
      <div style={{ width: 240, flexShrink: 0, display: "flex", flexDirection: "column", gap: 12 }}>
        <div className="card">
          <div className="card-title">Legend</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 14 }}>
            {(["green", "amber", "red"] as const).map((s) => (
              <div key={s} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ width: 12, height: 12, borderRadius: "50%", background: STATUS_COLOR[s], border: "2px solid var(--recessed)" }} />
                <span>{STATUS_LABEL[s]}</span>
              </div>
            ))}
          </div>
        </div>

        {selectedAsset && (
          <div className="card">
            <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 10 }}>{selectedAsset.name}</div>
            <div style={{ fontSize: 14, color: "var(--text-dim)", marginBottom: 4 }}>Type: {selectedAsset.type}</div>
            <div style={{ fontSize: 14, color: "var(--text-dim)", marginBottom: 4 }}>{selectedAsset.distance}m from flow path</div>
            {selectedAsset.pop > 0 && <div style={{ fontSize: 14, color: "var(--text-dim)", marginBottom: 4 }}>Population: {selectedAsset.pop}</div>}
            <div style={{ margin: "10px 0" }}>
              <span className={`badge ${STATUS_BADGE[selectedAsset.status]}`}>
                {selectedAsset.status === "green" ? "SAFE" : selectedAsset.status === "amber" ? "BUFFERED" : "INUNDATED"}
              </span>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                className="btn btn-ghost"
                style={{ fontSize: 13, padding: "6px 12px" }}
                onClick={() => mapRef.current?.flyTo({ center: [selectedAsset.lon, selectedAsset.lat], zoom: 14, duration: 1000 })}
              >
                Fly to
              </button>
              {onJumpToReview && (
                <button
                  className="btn btn-ghost"
                  style={{ fontSize: 13, padding: "6px 12px", color: "var(--accent)", borderColor: "var(--accent)" }}
                  onClick={onJumpToReview}
                >
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
