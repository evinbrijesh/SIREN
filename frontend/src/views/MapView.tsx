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
  { id: "village-2", type: "village", name: "Chhukung", lon: 86.86, lat: 27.9, status: "green", pop: 1240, distance: 210, buffer: 100 },
  { id: "BR-12", type: "bridge", name: "Hillary Bridge", lon: 86.85, lat: 27.91, status: "amber", pop: 0, distance: 60, buffer: 75 },
  { id: "RD-4", type: "road", name: "Road 4", lon: 86.86, lat: 27.9, status: "amber", pop: 0, distance: 40, buffer: 50 },
  { id: "well-3", type: "well", name: "Well 3", lon: 86.86, lat: 27.89, status: "red", pop: 0, distance: 90, buffer: 100 },
];

const STATUS_COLOR: Record<string, string> = {
  green: "#22C55E",
  amber: "#F59E0B",
  red: "#EF4444",
};

const STATUS_LABEL: Record<string, string> = {
  green: "Safe",
  amber: "Buffered",
  red: "Inundated",
};

const STEP_LABELS: Record<SimStep, string> = {
  before: "Baseline — Nov 22, 2025",
  "obs-1": "Obs 1 — Aug 23, 2026",
  "obs-2": "Obs 2 — Aug 29, 2026",
  "obs-3": "Obs 3 — Sep 4, 2026",
};

const LAYER_LABELS: { key: keyof MapViewLayers; label: string }[] = [
  { key: "basin", label: "Basin AOI" },
  { key: "optical", label: "Optical baseline" },
  { key: "sar", label: "SAR backscatter" },
  { key: "water", label: "Water expansion" },
  { key: "corridor", label: "D8 corridor" },
  { key: "assets", label: "OSM assets" },
];

type MapViewLayers = {
  basin: boolean;
  optical: boolean;
  sar: boolean;
  water: boolean;
  corridor: boolean;
  assets: boolean;
};

interface MapViewProps {
  onJumpToReview?: () => void;
}

export default function MapView({ onJumpToReview }: MapViewProps = {}) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markersRef = useRef<maplibregl.Marker[]>([]);
  const [swipePct, setSwipePct] = useState(50);
  const [opacity, setOpacity] = useState(100);
  const sim = useSimulation();

  const [layers, setLayers] = useState<MapViewLayers>({
    basin: true,
    optical: true,
    sar: false,
    water: true,
    corridor: true,
    assets: true,
  });

  const sarRevealed = sim.step === "obs-2" || sim.step === "obs-3";
  const selectedAsset = ASSET_MARKERS.find((a) => a.id === sim.selectedAssetId) ?? null;

  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {},
        layers: [{ id: "background", type: "background", paint: { "background-color": "#0A0F1E" } }],
      },
      center: [86.82, 27.88],
      zoom: 11,
    });
    mapRef.current = map;

    map.on("load", () => {
      map.addSource("basin", { type: "geojson", data: BASIN_POLYGON as any });
      map.addLayer({
        id: "basin-fill",
        type: "fill",
        source: "basin",
        paint: { "fill-color": "#06B6D4", "fill-opacity": 0.08 * (opacity / 100) },
        layout: { visibility: layers.basin ? "visible" : "none" },
      });
      map.addLayer({
        id: "basin-border",
        type: "line",
        source: "basin",
        paint: { "line-color": "#06B6D4", "line-width": 2 },
        layout: { visibility: layers.basin ? "visible" : "none" },
      });

      map.addSource("water", {
        type: "geojson",
        data: {
          type: "Feature",
          geometry: { type: "Polygon", coordinates: [[[86.8, 27.87], [86.84, 27.87], [86.84, 27.9], [86.8, 27.9], [86.8, 27.87]]] },
          properties: {},
        } as any,
      });
      map.addLayer({
        id: "water-fill",
        type: "fill",
        source: "water",
        paint: { "fill-color": "#3B82F6", "fill-opacity": 0.4 * (opacity / 100) },
        layout: { visibility: layers.water ? "visible" : "none" },
      });

      map.addSource("corridor", {
        type: "geojson",
        data: { type: "Feature", geometry: { type: "LineString", coordinates: [[86.82, 27.88], [86.85, 27.91]] }, properties: {} } as any,
      });
      map.addLayer({
        id: "corridor-line",
        type: "line",
        source: "corridor",
        paint: { "line-color": "#F59E0B", "line-width": 2, "line-dasharray": [6, 4] },
        layout: { visibility: layers.corridor ? "visible" : "none" },
      });

      ASSET_MARKERS.forEach((asset) => {
        const el = document.createElement("div");
        el.style.cssText = `width: 12px; height: 12px; border-radius: 50%; background: ${STATUS_COLOR[asset.status]}; border: 2px solid #0A0F1E; cursor: pointer;`;
        el.title = asset.name;
        el.onclick = () => sim.selectAsset(asset.id);
        const marker = new maplibregl.Marker(el).setLngLat([asset.lon, asset.lat]).addTo(map);
        markersRef.current.push(marker);
      });
    });

    return () => {
      map.remove();
      mapRef.current = null;
      markersRef.current = [];
    };
  }, []);

  useEffect(() => {
    if (selectedAsset && mapRef.current) {
      mapRef.current.flyTo({ center: [selectedAsset.lon, selectedAsset.lat], zoom: 13, duration: 1000 });
    }
  }, [sim.selectedAssetId]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (map.getLayer("basin-fill")) {
      map.setPaintProperty("basin-fill", "fill-opacity", 0.08 * (opacity / 100));
    }
    if (map.getLayer("water-fill")) {
      map.setPaintProperty("water-fill", "fill-opacity", 0.4 * (opacity / 100));
    }
  }, [opacity]);

  const toggleLayer = (key: keyof MapViewLayers) => {
    setLayers((prev) => {
      const next = { ...prev, [key]: !prev[key] };
      const map = mapRef.current;
      if (map) {
        if (key === "basin") {
          map.setLayoutProperty("basin-fill", "visibility", next.basin ? "visible" : "none");
          map.setLayoutProperty("basin-border", "visibility", next.basin ? "visible" : "none");
        }
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
    <div className="flex flex-row gap-space-12 h-full items-stretch">
      {/* Left dock */}
      <aside className="w-dock-left-width flex-none bg-surface-panel rounded-xl p-space-16 border border-border-subtle flex flex-col justify-between">
        <div className="space-y-space-16">
          <h2 className="text-headline-md font-medium text-text-primary tracking-tight">Layers</h2>
          <div className="flex flex-col space-y-space-8 text-body-md">
            {LAYER_LABELS.map(({ key, label }) => {
              const isLocked = key === "sar" && !sarRevealed;
              const checked = key === "sar" ? layers.sar && sarRevealed : layers[key];
              return (
                <label
                  key={key}
                  className={`flex items-center gap-space-8 select-none ${
                    isLocked ? "text-text-dim cursor-not-allowed opacity-60" : "text-text-primary cursor-pointer hover:text-primary transition-colors"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => !isLocked && toggleLayer(key)}
                    disabled={isLocked}
                    className="w-4 h-4 rounded bg-surface-recessed border border-border-subtle text-primary focus:ring-0 focus:ring-offset-0 cursor-pointer disabled:cursor-not-allowed disabled:opacity-40"
                  />
                  <span>{label}</span>
                  {isLocked && <span className="text-caption text-text-dim">(locked)</span>}
                </label>
              );
            })}
          </div>
        </div>
        <div className="pt-space-16 border-t border-border-subtle">
          <div className="flex justify-between items-center mb-space-8">
            <span className="text-body-sm text-text-dim uppercase tracking-wider">Opacity</span>
            <span className="font-mono text-code-sm text-primary-container">{opacity}%</span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={opacity}
            onChange={(e) => setOpacity(parseInt(e.target.value, 10))}
            className="w-full h-1 rounded bg-surface-recessed appearance-none cursor-pointer accent-primary-container"
          />
        </div>
      </aside>

      {/* Center */}
      <div className="flex-1 flex flex-col gap-space-12 min-w-0">
        <div className="relative flex-1 min-h-0 bg-surface-recessed rounded-xl overflow-hidden border border-border-subtle">
          <div ref={mapContainer} className="absolute inset-0" />
          <div className="absolute top-space-16 left-space-16 z-10 px-space-12 py-space-6 bg-surface-panel rounded border border-border-subtle text-body-sm text-text-primary">
            {STEP_LABELS[sim.step]}
          </div>
        </div>

        <section className="bg-surface-panel rounded-xl p-space-16 border border-border-subtle flex flex-col gap-space-12 flex-none">
          <div className="flex justify-between items-center">
            <h3 className="text-headline-md font-medium text-text-primary tracking-tight">Before / After</h3>
            <span className="text-caption text-text-dim">Drag vertical divider to compare expansion limits</span>
          </div>
          <div
            className="relative w-full h-[100px] rounded bg-surface-recessed overflow-hidden select-none cursor-ew-resize border border-border-subtle"
            onMouseMove={handleSwipe}
            onTouchMove={handleSwipe}
          >
            <div className="absolute inset-0 bg-[#00424f]/40 flex items-center justify-end px-space-24">
              <span className="text-body-sm text-primary-container font-medium">After — 4.1 km² (+14.3%)</span>
            </div>
            <div
              className="absolute inset-y-0 left-0 bg-surface-recessed flex items-center px-space-24 overflow-hidden border-r-0"
              style={{ width: `${swipePct}%` }}
            >
              <span className="text-body-sm text-text-dim shrink-0">Before — 3.0 km²</span>
            </div>
            <div
              className="absolute top-0 bottom-0 z-30 pointer-events-none flex items-center justify-center -ml-[1px]"
              style={{ left: `${swipePct}%` }}
            >
              <div className="w-[2px] h-full bg-primary-container" />
              <div className="absolute w-5 h-5 rounded-full bg-surface-panel flex items-center justify-center border-[1.5px] border-primary-container shadow-md">
                <span className="text-[10px] text-primary-container font-bold leading-none select-none">‹›</span>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* Right dock */}
      <aside className="w-dock-right-width flex-none bg-surface-panel rounded-xl p-space-16 border border-border-subtle flex flex-col justify-between">
        <div className="space-y-space-24">
          <div className="space-y-space-12">
            <h2 className="text-headline-md font-medium text-text-primary tracking-tight">Legend</h2>
            <div className="flex flex-col space-y-space-8 text-body-md">
              {([
                ["green", "Safe"],
                ["amber", "Buffered"],
                ["red", "Inundated"],
              ] as const).map(([status, label]) => (
                <div key={status} className="flex items-center gap-space-8">
                  <span
                    className="w-3 h-3 rounded-full shrink-0"
                    style={{ backgroundColor: STATUS_COLOR[status], border: "2px solid #0A0F1E" }}
                  />
                  <span className="text-text-primary">{label}</span>
                </div>
              ))}
            </div>
          </div>

          {selectedAsset && (
            <div className="pt-space-16 border-t border-border-subtle space-y-space-12">
              <div className="space-y-space-4">
                <span className="text-caption text-text-dim uppercase tracking-wider">Target Inspection</span>
                <h3 className="text-headline-md font-medium text-text-primary">{selectedAsset.name}</h3>
              </div>
              <div className="space-y-space-4 text-body-md text-text-dim">
                <p>Type: {selectedAsset.type}</p>
                <p>Distance: {selectedAsset.distance}m from corridor</p>
                {selectedAsset.pop > 0 && <p>Population: {selectedAsset.pop.toLocaleString()}</p>}
              </div>
              <div className="pt-space-4">
                <span
                  className="inline-block px-space-8 py-space-2 rounded text-body-sm tracking-wider uppercase bg-transparent border"
                  style={{
                    color: STATUS_COLOR[selectedAsset.status],
                    borderColor: STATUS_COLOR[selectedAsset.status],
                  }}
                >
                  {STATUS_LABEL[selectedAsset.status].toUpperCase()}
                </span>
              </div>
              <div className="pt-space-12 flex items-center justify-between">
                <button
                  onClick={() => mapRef.current?.flyTo({ center: [selectedAsset.lon, selectedAsset.lat], zoom: 14, duration: 1000 })}
                  className="px-space-12 py-space-6 rounded text-body-sm text-text-dim hover:text-text-primary transition-colors bg-transparent border border-border-subtle hover:border-text-dim"
                >
                  Fly to
                </button>
                {onJumpToReview && (
                  <button
                    onClick={onJumpToReview}
                    className="px-space-12 py-space-6 rounded text-body-sm text-primary border border-primary hover:bg-surface-recessed transition-colors bg-transparent"
                  >
                    Review →
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
