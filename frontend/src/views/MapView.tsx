import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { api, apiOrMock } from "../api/client";
import { mockData } from "../api/mockData";
import { useSimulation, type SimStep } from "../simulation/SimulationContext";
import type { BasinConfig, Run, ExposureList, MlEvidence, GeoJSONFeature } from "../api/types";

interface MapViewProps {
  basin?: BasinConfig;
  run?: Run;
  onJumpToReview?: () => void;
}

const STEP_LABELS: Record<SimStep, string> = {
  before: "Baseline — Nov 22, 2025",
  "obs-1": "Obs 1 — Aug 23, 2026",
  "obs-2": "Obs 2 — Aug 29, 2026",
  "obs-3": "Obs 3 — Sep 4, 2026",
};

const STATUS_COLOR: Record<string, string> = {
  safe: "var(--color-status-safe)",
  buffered: "var(--color-status-warn)",
  amber: "var(--color-status-warn)",
  inundated: "var(--color-status-danger)",
  red: "var(--color-status-danger)",
  green: "var(--color-status-safe)",
};

const STATUS_LABEL: Record<string, string> = {
  safe: "Safe",
  buffered: "Buffered",
  amber: "Buffered",
  inundated: "Inundated",
  red: "Inundated",
  green: "Safe",
};

const LAYER_ORDER = ["basemap", "basin-fill", "basin-border", "water-fill", "water-outline", "corridor-line", "heatmap-overlay"];

type MapViewLayers = {
  basemap: boolean;
  basin: boolean;
  water: boolean;
  corridor: boolean;
  assets: boolean;
  heatmap: boolean;
};

const LAYER_LABELS: { key: keyof MapViewLayers; label: string; icon: string }[] = [
  { key: "basemap", label: "Satellite basemap", icon: "🛰️" },
  { key: "basin", label: "Basin AOI", icon: "□" },
  { key: "water", label: "Water expansion", icon: "💧" },
  { key: "corridor", label: "D8 corridor", icon: "→" },
  { key: "assets", label: "OSM assets", icon: "📍" },
  { key: "heatmap", label: "ML heatmap", icon: "🔥" },
];

function isFeatureCollection(g: GeoJSONFeature | null): g is { type: "FeatureCollection"; features: any[] } {
  return g?.type === "FeatureCollection";
}

function isLineString(g: GeoJSONFeature | null): g is { type: "LineString"; coordinates: any } {
  return g?.type === "LineString";
}

function isPolygon(g: GeoJSONFeature | null): g is { type: "Polygon"; coordinates: any } {
  return g?.type === "Polygon";
}

function toFeature(geom: GeoJSONFeature | null): any {
  if (!geom) return { type: "Feature", geometry: { type: "Point", coordinates: [] }, properties: {} };
  if (geom.type === "Feature") return geom;
  if (geom.type === "FeatureCollection") {
    const first = geom.features?.[0]?.geometry;
    return { type: "Feature", geometry: first || { type: "Point", coordinates: [] }, properties: {} };
  }
  return { type: "Feature", geometry: geom, properties: {} };
}

export default function MapView({ basin, run, onJumpToReview }: MapViewProps = {}) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markersRef = useRef<maplibregl.Marker[]>([]);
  const swipeContainerRef = useRef<HTMLDivElement>(null);
  const [swipePct, setSwipePct] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const [opacity, setOpacity] = useState(100);
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);
  const sim = useSimulation();

  const [layers, setLayers] = useState<MapViewLayers>({
    basemap: true,
    basin: false,
    water: true,
    corridor: true,
    assets: true,
    heatmap: false,
  });

  const runId = run?.run_id;

  const { data: exposuresData } = useQuery({
    queryKey: ["exposures", runId],
    queryFn: () => (runId ? apiOrMock(() => api.listExposures(runId), "exposures") as Promise<ExposureList> : Promise.resolve(mockData.exposures)),
    enabled: !!runId,
  });

  const { data: mlData } = useQuery({
    queryKey: ["ml-evidence", runId],
    queryFn: () => (runId ? apiOrMock(() => api.getMlEvidence(runId), "mlEvidence") as Promise<MlEvidence> : Promise.resolve(mockData.mlEvidence)),
    enabled: !!runId,
  });

  const exposures = exposuresData?.exposures ?? mockData.exposures.exposures;
  const mlEvidence = mlData ?? mockData.mlEvidence;

  const basinPolygon = basin?.boundary_geojson ?? mockData.basin.boundary_geojson;
  const corridorGeojson = run?.corridor_geojson ?? mockData.runs.runs[0].corridor_geojson;
  const changePolygon = mlEvidence.change_polygon ?? (run?.change_stats_json?.change_polygon as GeoJSONFeature | undefined) ?? null;
  const basemapUri = basin?.basemap_uri;
  const basemapBounds = basin?.basemap_bounds;

  const beforeArea = typeof run?.change_stats_json?.water_area_km2 === "number" ? (run.change_stats_json.water_area_km2 as number) / (1 + ((run.change_stats_json.expansion_percent as number) / 100)) : 3.0;
  const afterArea = typeof run?.change_stats_json?.water_area_km2 === "number" ? run.change_stats_json.water_area_km2 : 4.1;
  const expansionPct = typeof run?.change_stats_json?.expansion_percent === "number" ? run.change_stats_json.expansion_percent : 14.3;

  // Initialize map
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
      // Basemap satellite image
      if (basemapUri && basemapBounds?.length === 4) {
        map.addSource("basemap", {
          type: "image",
          url: basemapUri,
          coordinates: basemapBounds as [[number, number], [number, number], [number, number], [number, number]],
        });
        map.addLayer({
          id: "basemap",
          type: "raster",
          source: "basemap",
          paint: { "raster-opacity": layers.basemap ? 1 : 0, "raster-fade-duration": 0 },
        });
      }

      // Basin AOI
      map.addSource("basin", { type: "geojson", data: basinPolygon as any });
      map.addLayer({
        id: "basin-fill",
        type: "fill",
        source: "basin",
        paint: { "fill-color": "var(--color-primary)", "fill-opacity": 0.05 * (opacity / 100) },
        layout: { visibility: layers.basin ? "visible" : "none" },
      });
      map.addLayer({
        id: "basin-border",
        type: "line",
        source: "basin",
        paint: { "line-color": "var(--color-primary)", "line-width": 1.5, "line-dasharray": [4, 4] },
        layout: { visibility: layers.basin ? "visible" : "none" },
      });

      // Water expansion polygon
      map.addSource("water", { type: "geojson", data: toFeature(changePolygon) as any });
      map.addLayer({
        id: "water-fill",
        type: "fill",
        source: "water",
        paint: { "fill-color": "var(--color-status-danger)", "fill-opacity": 0.35 * (opacity / 100) },
        layout: { visibility: layers.water ? "visible" : "none" },
      });
      map.addLayer({
        id: "water-outline",
        type: "line",
        source: "water",
        paint: { "line-color": "var(--color-status-danger)", "line-width": 1.5 },
        layout: { visibility: layers.water ? "visible" : "none" },
      });

      // Corridor
      const corridorFeature = toFeature(corridorGeojson);
      map.addSource("corridor", { type: "geojson", data: corridorFeature as any });
      map.addLayer({
        id: "corridor-line",
        type: "line",
        source: "corridor",
        paint: { "line-color": "var(--color-status-warn)", "line-width": 2.5, "line-dasharray": [6, 4] },
        layout: { visibility: layers.corridor ? "visible" : "none" },
      });

      // Assets
      updateAssetMarkers(map, exposures, markersRef, sim.selectAsset);

      // Heatmap overlay
      const heatBounds = mlEvidence.heatmap_bounds ?? mlEvidence.mask_bounds;
      if (heatBounds?.length === 4) {
        map.addSource("heatmap", {
          type: "image",
          url: mlEvidence.heatmap_uri,
          coordinates: heatBounds as [[number, number], [number, number], [number, number], [number, number]],
        });
        map.addLayer({
          id: "heatmap-overlay",
          type: "raster",
          source: "heatmap",
          paint: { "raster-opacity": 0.65 * (opacity / 100), "raster-fade-duration": 0 },
          layout: { visibility: layers.heatmap ? "visible" : "none" },
        });
      }
    });

    return () => {
      map.remove();
      mapRef.current = null;
      markersRef.current = [];
    };
  }, []);

  // Update sources when data changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (map.getSource("basin") && basinPolygon) {
      (map.getSource("basin") as any).setData(basinPolygon);
    }
    if (map.getSource("water") && changePolygon) {
      (map.getSource("water") as any).setData(toFeature(changePolygon));
    }
    if (map.getSource("corridor") && corridorGeojson) {
      (map.getSource("corridor") as any).setData(toFeature(corridorGeojson));
    }
    if (map.getSource("basemap") && basemapUri && basemapBounds?.length === 4) {
      (map.getSource("basemap") as any).updateImage({ url: basemapUri, coordinates: basemapBounds as [[number, number], [number, number], [number, number], [number, number]] });
    }
    updateAssetMarkers(map, exposures, markersRef, sim.selectAsset);

    const heatBounds = mlEvidence.heatmap_bounds ?? mlEvidence.mask_bounds;
    if (heatBounds?.length === 4 && map.getSource("heatmap")) {
      (map.getSource("heatmap") as any).updateImage({
        url: mlEvidence.heatmap_uri,
        coordinates: heatBounds as [[number, number], [number, number], [number, number], [number, number]],
      });
    }
  }, [basinPolygon, corridorGeojson, changePolygon, basemapUri, basemapBounds, exposures, mlEvidence]);

  // Update layer visibility + opacity
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const setVis = (id: string, visible: boolean) => {
      if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
    };
    const setOp = (id: string, prop: string, base: number) => {
      if (map.getLayer(id)) map.setPaintProperty(id, prop, base * (opacity / 100));
    };

    setVis("basemap", layers.basemap);
    setVis("basin-fill", layers.basin);
    setVis("basin-border", layers.basin);
    setVis("water-fill", layers.water);
    setVis("water-outline", layers.water);
    setVis("corridor-line", layers.corridor);
    setVis("heatmap-overlay", layers.heatmap);

    setOp("basin-fill", "fill-opacity", 0.05);
    setOp("water-fill", "fill-opacity", 0.35);
    setOp("heatmap-overlay", "raster-opacity", 0.65);
    setOp("basemap", "raster-opacity", 1);
  }, [layers, opacity]);

  // Fly to selected asset
  useEffect(() => {
    if (!sim.selectedAssetId || !mapRef.current) return;
    const coords = ASSET_COORDS[sim.selectedAssetId];
    if (coords) mapRef.current.flyTo({ center: coords, zoom: 14, duration: 1000 });
  }, [sim.selectedAssetId]);

  const toggleLayer = (key: keyof MapViewLayers) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const updateSwipeFromEvent = (clientX: number) => {
    const container = swipeContainerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    setSwipePct(pct);
  };

  const startDrag = (e: React.MouseEvent | React.TouchEvent) => {
    setIsDragging(true);
    const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
    updateSwipeFromEvent(clientX);
    e.preventDefault();
  };

  useEffect(() => {
    if (!isDragging) return;
    const onMove = (e: MouseEvent | TouchEvent) => {
      const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
      updateSwipeFromEvent(clientX);
    };
    const onUp = () => setIsDragging(false);
    window.addEventListener("mousemove", onMove as EventListener);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchmove", onMove as EventListener, { passive: true });
    window.addEventListener("touchend", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove as EventListener);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchmove", onMove as EventListener);
      window.removeEventListener("touchend", onUp);
    };
  }, [isDragging]);

  const beforeImg = mlEvidence.preview_baseline_uri || mlEvidence.baseline_mask_uri;
  const afterImg = mlEvidence.preview_after_uri || mlEvidence.heatmap_uri || mlEvidence.mask_uri;

  const selectedAsset = exposures.find((e) => e.asset_id === sim.selectedAssetId);

  return (
    <div className="flex flex-row gap-space-12 h-full items-stretch">
      {/* Left dock */}
      <aside className={`${leftOpen ? "w-dock-left-width" : "w-10"} flex-none bg-surface-panel rounded-xl p-space-16 border border-border-subtle flex flex-col justify-between transition-all`}>
        {leftOpen ? (
          <>
            <div className="space-y-space-16">
              <div className="flex items-center justify-between">
                <h2 className="text-headline-md font-headline text-text-primary tracking-tight">Layers</h2>
                <button onClick={() => setLeftOpen(false)} className="text-text-dim hover:text-text-primary">◀</button>
              </div>
              <div className="flex flex-col space-y-space-10 text-body-md">
                {LAYER_LABELS.map(({ key, label, icon }) => (
                  <label
                    key={key}
                    className="flex items-center gap-space-8 select-none text-text-primary cursor-pointer hover:text-primary transition-colors"
                  >
                    <input
                      type="checkbox"
                      checked={layers[key]}
                      onChange={() => toggleLayer(key)}
                      className="w-4 h-4 rounded bg-surface-recessed border border-border-subtle text-primary focus:ring-0 focus:ring-offset-0 cursor-pointer"
                    />
                    <span className="text-body-sm">{icon}</span>
                    <span>{label}</span>
                  </label>
                ))}
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
                className="w-full h-1.5 rounded bg-surface-recessed appearance-none cursor-pointer accent-primary-container"
              />
            </div>
          </>
        ) : (
          <button onClick={() => setLeftOpen(true)} className="text-text-dim hover:text-text-primary mx-auto">▶</button>
        )}
      </aside>

      {/* Center */}
      <div className="flex-1 flex flex-col gap-space-12 min-w-0">
        <div className="relative flex-1 min-h-0 bg-surface-recessed rounded-xl overflow-hidden border border-border-subtle shadow-panel">
          <div ref={mapContainer} className="absolute inset-0" />
          <div className="absolute top-space-16 left-space-16 z-10 px-space-12 py-space-6 bg-surface-panel/90 backdrop-blur rounded border border-border-subtle text-body-sm text-text-primary shadow-panel">
            {STEP_LABELS[sim.step]}
          </div>
        </div>

        <section className="bg-surface-panel rounded-xl p-space-16 border border-border-subtle flex flex-col gap-space-12 flex-none shadow-panel">
          <div className="flex justify-between items-center">
            <h3 className="text-headline-md font-headline text-text-primary tracking-tight">Before / After</h3>
            <span className="text-caption text-text-dim">Drag vertical divider to compare expansion limits</span>
          </div>
          <div
            ref={swipeContainerRef}
            className="relative w-full h-[160px] rounded bg-surface-recessed overflow-hidden select-none cursor-ew-resize border border-border-subtle"
            onMouseDown={startDrag}
            onTouchStart={startDrag}
          >
            {/* After image (full, visible where not clipped) */}
            <div className="absolute inset-0 bg-surface-canvas flex items-center justify-center">
              {afterImg ? (
                <img src={afterImg} alt="after" className="h-full w-full object-cover" />
              ) : (
                <span className="text-body-sm text-text-dim">After — no image</span>
              )}
            </div>
            {/* Before image, clipped to swipe width; image is same rendered size as after */}
            <div
              className="absolute inset-y-0 left-0 bg-surface-canvas overflow-hidden flex items-center justify-center"
              style={{ width: `${swipePct}%` }}
            >
              {beforeImg ? (
                <img
                  src={beforeImg}
                  alt="before"
                  className="h-full w-full object-cover"
                  style={{ minWidth: `${100 / (swipePct / 100 || 1)}%` }}
                />
              ) : (
                <span className="text-body-sm text-text-dim shrink-0">Before — no image</span>
              )}
            </div>
            <div className="absolute top-2 left-2 text-body-xs text-text-dim bg-surface-panel/90 px-1.5 rounded">
              Before — {beforeArea.toFixed(1)} km²
            </div>
            <div className="absolute top-2 right-2 text-body-xs text-primary-container bg-surface-panel/90 px-1.5 rounded">
              After — {afterArea.toFixed(1)} km² (+{expansionPct.toFixed(1)}%)
            </div>
            <div
              className="absolute top-0 bottom-0 z-30 pointer-events-none flex items-center justify-center -ml-[2px]"
              style={{ left: `${swipePct}%` }}
            >
              <div className="w-[3px] h-full bg-primary-container" />
              <div className="absolute w-6 h-6 rounded-full bg-surface-panel flex items-center justify-center border-2 border-primary-container shadow-md">
                <span className="text-[10px] text-primary-container font-bold leading-none select-none">‹›</span>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* Right dock */}
      <aside className={`${rightOpen ? "w-dock-right-width" : "w-10"} flex-none bg-surface-panel rounded-xl p-space-16 border border-border-subtle flex flex-col justify-between transition-all`}>
        {rightOpen ? (
          <>
            <div className="flex items-center justify-between mb-space-12">
              <h2 className="text-headline-md font-headline text-text-primary tracking-tight">Legend</h2>
              <button onClick={() => setRightOpen(false)} className="text-text-dim hover:text-text-primary">▶</button>
            </div>
            <div className="space-y-space-24">
              <div className="space-y-space-12">
                <div className="flex flex-col space-y-space-8 text-body-md">
                  {([
                    ["green", "Safe"],
                    ["amber", "Buffered"],
                    ["red", "Inundated"],
                  ] as const).map(([status, label]) => (
                    <div key={status} className="flex items-center gap-space-8">
                      <span
                        className="w-3 h-3 rounded-full shrink-0"
                        style={{ backgroundColor: `var(--color-status-${status === "green" ? "safe" : status === "amber" ? "warn" : "danger"})`, border: "2px solid var(--color-surface-canvas)" }}
                      />
                      <span className="text-text-primary">{label}</span>
                    </div>
                  ))}
                </div>
              </div>

              {selectedAsset && (
                <div className="pt-space-16 border-t border-border-subtle space-y-space-12">
                  <AssetDetail asset={selectedAsset} onJumpToReview={onJumpToReview} />
                </div>
              )}
            </div>
          </>
        ) : (
          <button onClick={() => setRightOpen(true)} className="text-text-dim hover:text-text-primary mx-auto">◀</button>
        )}
      </aside>
    </div>
  );
}

const ASSET_COORDS: Record<string, [number, number]> = {
  "village-2": [86.871, 27.904],
  "BR-12": [86.852, 27.913],
  "RD-4": [86.863, 27.907],
  "well-3": [86.861, 27.899],
};

function updateAssetMarkers(map: maplibregl.Map, exposures: any[], markersRef: React.MutableRefObject<maplibregl.Marker[]>, selectAsset: (id: string | null) => void) {
  markersRef.current.forEach((m) => m.remove());
  markersRef.current = [];

  exposures.forEach((asset) => {
    const coords = asset.lon != null && asset.lat != null ? ([asset.lon, asset.lat] as [number, number]) : ASSET_COORDS[asset.asset_id];
    if (!coords) return;
    const status = asset.inundated ? "red" : asset.distance_m != null ? "amber" : "green";
    const el = document.createElement("div");
    el.style.cssText = `width: 14px; height: 14px; border-radius: 50%; background: ${STATUS_COLOR[status]}; border: 2px solid var(--color-surface-canvas); cursor: pointer; box-shadow: 0 0 0 1px var(--color-border-subtle);`;
    el.title = asset.name || asset.asset_id;
    el.onclick = () => selectAsset(asset.asset_id);
    const marker = new maplibregl.Marker(el).setLngLat(coords).addTo(map);
    markersRef.current.push(marker);
  });
}

function AssetDetail({ asset, onJumpToReview }: { asset: any; onJumpToReview?: () => void }) {
  const status = asset.inundated ? "red" : asset.distance_m != null ? "amber" : "green";
  return (
    <div className="space-y-space-12">
      <div className="space-y-space-4">
        <span className="text-caption text-text-dim uppercase tracking-wider">Target Inspection</span>
        <h3 className="text-headline-md font-headline text-text-primary">{asset.name || asset.asset_id}</h3>
      </div>
      <div className="space-y-space-4 text-body-md text-text-dim">
        <p>Type: {asset.asset_type}</p>
        <p>Distance: {asset.distance_m}m from corridor</p>
        {asset.population != null && asset.population > 0 && <p>Population: {asset.population.toLocaleString()}</p>}
      </div>
      <span
        className="inline-block px-space-8 py-space-2 rounded text-body-sm tracking-wider uppercase bg-transparent border"
        style={{ color: STATUS_COLOR[status], borderColor: STATUS_COLOR[status] }}
      >
        {STATUS_LABEL[status].toUpperCase()}
      </span>
      {onJumpToReview && (
        <button
          onClick={onJumpToReview}
          className="w-full mt-space-8 px-space-12 py-space-6 rounded text-body-sm text-primary border border-primary hover:bg-surface-recessed transition-colors bg-transparent"
        >
          Review →
        </button>
      )}
    </div>
  );
}
