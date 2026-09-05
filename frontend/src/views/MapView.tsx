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
  before: "BASELINE | 2025-11-22",
  "obs-1": "OBS 01 | 2026-08-23",
  "obs-2": "OBS 02 | 2026-08-29",
  "obs-3": "OBS 03 | 2026-09-04",
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
  safe: "SAFE",
  buffered: "BUFFERED",
  amber: "BUFFERED",
  inundated: "INUNDATED",
  red: "INUNDATED",
  green: "SAFE",
};

type MapViewLayers = {
  basemap: boolean;
  basin: boolean;
  water: boolean;
  corridor: boolean;
  assets: boolean;
  heatmap: boolean;
};

const LAYER_LABELS: { key: keyof MapViewLayers; label: string }[] = [
  { key: "basemap", label: "Satellite basemap" },
  { key: "basin", label: "Basin AOI" },
  { key: "water", label: "Water expansion" },
  { key: "corridor", label: "D8 corridor" },
  { key: "assets", label: "OSM assets" },
  { key: "heatmap", label: "ML heatmap" },
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
        layers: [{ id: "background", type: "background", paint: { "background-color": "#0b0f17" } }],
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
        paint: { "fill-color": "#38bdf8", "fill-opacity": 0.05 * (opacity / 100) },
        layout: { visibility: layers.basin ? "visible" : "none" },
      });
      map.addLayer({
        id: "basin-border",
        type: "line",
        source: "basin",
        paint: { "line-color": "#38bdf8", "line-width": 1, "line-dasharray": [4, 4] },
        layout: { visibility: layers.basin ? "visible" : "none" },
      });

      // Water expansion polygon
      map.addSource("water", { type: "geojson", data: toFeature(changePolygon) as any });
      map.addLayer({
        id: "water-fill",
        type: "fill",
        source: "water",
        paint: { "fill-color": "#ef4444", "fill-opacity": 0.35 * (opacity / 100) },
        layout: { visibility: layers.water ? "visible" : "none" },
      });
      map.addLayer({
        id: "water-outline",
        type: "line",
        source: "water",
        paint: { "line-color": "#ef4444", "line-width": 1.5 },
        layout: { visibility: layers.water ? "visible" : "none" },
      });

      // Corridor
      const corridorFeature = toFeature(corridorGeojson);
      map.addSource("corridor", { type: "geojson", data: corridorFeature as any });
      map.addLayer({
        id: "corridor-line",
        type: "line",
        source: "corridor",
        paint: { "line-color": "#f59e0b", "line-width": 2, "line-dasharray": [6, 4] },
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
    <div className="flex flex-row h-full items-stretch">
      {/* Left dock — layers */}
      <aside className={`${leftOpen ? "w-dock-left-width" : "w-8"} flex-none bg-surface-panel border-r border-border-subtle flex flex-col justify-between transition-all`}>
        {leftOpen ? (
          <>
            <div className="p-space-12 space-y-space-12">
              <div className="flex items-center justify-between">
                <h2 className="label-caps">Layers</h2>
                <button onClick={() => setLeftOpen(false)} className="text-text-dim hover:text-text-primary text-body-sm">[&lt;]</button>
              </div>
              <div className="flex flex-col space-y-space-2">
                {LAYER_LABELS.map(({ key, label }) => (
                  <label
                    key={key}
                    className="flex items-center gap-space-8 select-none text-text-primary cursor-pointer hover:bg-surface-container px-space-4 py-space-2 transition-colors"
                  >
                    <input
                      type="checkbox"
                      checked={layers[key]}
                      onChange={() => toggleLayer(key)}
                      className="w-3 h-3 bg-surface-recessed border border-border-strong text-primary focus:ring-0 focus:ring-offset-0 cursor-pointer"
                    />
                    <span className="text-body-md">{label}</span>
                  </label>
                ))}
              </div>
            </div>
            <div className="p-space-12 border-t border-border-subtle">
              <div className="flex justify-between items-center mb-space-4">
                <span className="label-caps">Overlay Opacity</span>
                <span className="data-val text-body-sm text-text-primary">{opacity}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                value={opacity}
                onChange={(e) => setOpacity(parseInt(e.target.value, 10))}
                className="w-full h-1 bg-surface-recessed appearance-none cursor-pointer accent-primary-container"
              />
            </div>
          </>
        ) : (
          <button onClick={() => setLeftOpen(true)} className="text-text-dim hover:text-text-primary mx-auto py-space-8 text-body-sm">[&gt;]</button>
        )}
      </aside>

      {/* Center — map + before/after */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="relative flex-1 min-h-0 bg-surface-recessed overflow-hidden border-b border-border-subtle">
          <div ref={mapContainer} className="absolute inset-0" />
          <div className="absolute top-space-8 left-space-8 z-10 px-space-8 py-space-4 bg-surface-panel border border-border-subtle data-val text-body-sm text-text-primary">
            {STEP_LABELS[sim.step]}
          </div>
          <div className="absolute top-space-8 right-space-8 z-10 px-space-8 py-space-4 bg-surface-panel border border-border-subtle data-val text-body-sm text-text-dim">
            27.88N 86.82E | UTM 45N
          </div>
        </div>

        <section className="bg-surface-panel border-t border-border-subtle flex flex-col flex-none">
          <div className="flex justify-between items-center px-space-12 py-space-8 border-b border-border-subtle">
            <h3 className="label-caps">Before / After — Swipe Compare</h3>
            <span className="data-val text-body-sm text-text-dim">DRAG DIVIDER</span>
          </div>
          <div
            ref={swipeContainerRef}
            className="relative w-full h-[140px] bg-surface-recessed overflow-hidden select-none cursor-ew-resize"
            onMouseDown={startDrag}
            onTouchStart={startDrag}
          >
            {/* After image (full, visible where not clipped) */}
            <div className="absolute inset-0 bg-surface-canvas flex items-center justify-center">
              {afterImg ? (
                <img src={afterImg} alt="after" className="h-full w-full object-cover" />
              ) : (
                <span className="data-val text-body-sm text-text-muted">N/A — no image</span>
              )}
            </div>
            {/* Before image, clipped to swipe width */}
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
                <span className="data-val text-body-sm text-text-muted shrink-0">N/A — no image</span>
              )}
            </div>
            <div className="absolute top-space-4 left-space-4 data-val text-caption text-text-dim bg-surface-panel border border-border-subtle px-space-4 py-space-2">
              BEFORE | {beforeArea.toFixed(2)} km²
            </div>
            <div className="absolute top-space-4 right-space-4 data-val text-caption text-status-elevated bg-surface-panel border border-border-subtle px-space-4 py-space-2">
              AFTER | {afterArea.toFixed(2)} km² | Δ +{expansionPct.toFixed(1)}%
            </div>
            <div
              className="absolute top-0 bottom-0 z-30 pointer-events-none flex items-center justify-center -ml-[1px]"
              style={{ left: `${swipePct}%` }}
            >
              <div className="w-[2px] h-full bg-primary-container" />
              <div className="absolute w-4 h-4 bg-surface-panel flex items-center justify-center border border-primary-container">
                <span className="text-[9px] text-primary-container font-bold leading-none select-none data-val">&lt;&gt;</span>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* Right dock — legend + asset detail */}
      <aside className={`${rightOpen ? "w-dock-right-width" : "w-8"} flex-none bg-surface-panel border-l border-border-subtle flex flex-col transition-all`}>
        {rightOpen ? (
          <>
            <div className="p-space-12 border-b border-border-subtle">
              <div className="flex items-center justify-between mb-space-8">
                <h2 className="label-caps">Legend</h2>
                <button onClick={() => setRightOpen(false)} className="text-text-dim hover:text-text-primary text-body-sm">[&gt;]</button>
              </div>
              <div className="flex flex-col space-y-space-4">
                {([
                  ["green", "SAFE"],
                  ["amber", "BUFFERED"],
                  ["red", "INUNDATED"],
                ] as const).map(([status, label]) => (
                  <div key={status} className="flex items-center gap-space-8">
                    <span
                      className="w-2.5 h-2.5 shrink-0"
                      style={{ backgroundColor: `var(--color-status-${status === "green" ? "safe" : status === "amber" ? "warn" : "danger"})` }}
                    />
                    <span className="text-body-md text-text-primary data-val">{label}</span>
                  </div>
                ))}
              </div>
            </div>

            {selectedAsset && (
              <div className="p-space-12 flex-1 overflow-auto">
                <AssetDetail asset={selectedAsset} onJumpToReview={onJumpToReview} />
              </div>
            )}
          </>
        ) : (
          <button onClick={() => setRightOpen(true)} className="text-text-dim hover:text-text-primary mx-auto py-space-8 text-body-sm">[&lt;]</button>
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
    el.style.cssText = `width: 10px; height: 10px; background: ${STATUS_COLOR[status]}; border: 1px solid #0b0f17; cursor: pointer;`;
    el.title = asset.name || asset.asset_id;
    el.onclick = () => selectAsset(asset.asset_id);
    const marker = new maplibregl.Marker(el).setLngLat(coords).addTo(map);
    markersRef.current.push(marker);
  });
}

function AssetDetail({ asset, onJumpToReview }: { asset: any; onJumpToReview?: () => void }) {
  const status = asset.inundated ? "red" : asset.distance_m != null ? "amber" : "green";
  return (
    <div className="space-y-space-8">
      <div className="space-y-space-2">
        <span className="label-caps">Target Inspection</span>
        <h3 className="text-headline-md text-text-primary data-val">{asset.name || asset.asset_id}</h3>
      </div>
      <div className="space-y-space-2 text-body-md text-text-dim data-val">
        <div>TYPE: {asset.asset_type}</div>
        <div>DIST: {asset.distance_m != null ? `${asset.distance_m.toFixed(0)} m` : "N/A"}</div>
        {asset.population != null && asset.population > 0 && <div>POP: {asset.population.toLocaleString()}</div>}
      </div>
      <span
        className="inline-block px-space-8 py-space-2 data-val text-body-sm tracking-wide border"
        style={{ color: STATUS_COLOR[status], borderColor: STATUS_COLOR[status] }}
      >
        {STATUS_LABEL[status]}
      </span>
      {onJumpToReview && (
        <button
          onClick={onJumpToReview}
          className="w-full mt-space-4 px-space-8 py-space-4 data-val text-body-sm text-primary border border-primary hover:bg-surface-container transition-colors bg-transparent"
        >
          REVIEW &gt;
        </button>
      )}
    </div>
  );
}
