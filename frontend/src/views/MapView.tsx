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
  safe: "#22C55E",
  buffered: "#F59E0B",
  amber: "#F59E0B",
  inundated: "#EF4444",
  red: "#EF4444",
  green: "#22C55E",
};

const STATUS_LABEL: Record<string, string> = {
  safe: "Safe",
  buffered: "Buffered",
  amber: "Buffered",
  inundated: "Inundated",
  red: "Inundated",
  green: "Safe",
};

type MapViewLayers = {
  basin: boolean;
  optical: boolean;
  sar: boolean;
  water: boolean;
  corridor: boolean;
  assets: boolean;
};

const LAYER_LABELS: { key: keyof MapViewLayers; label: string }[] = [
  { key: "basin", label: "Basin AOI" },
  { key: "optical", label: "Optical baseline" },
  { key: "sar", label: "SAR backscatter" },
  { key: "water", label: "Water expansion" },
  { key: "corridor", label: "D8 corridor" },
  { key: "assets", label: "OSM assets" },
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

export default function MapView({ basin, run, onJumpToReview }: MapViewProps = {}) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markersRef = useRef<maplibregl.Marker[]>([]);
  const [swipePct, setSwipePct] = useState(50);
  const [opacity, setOpacity] = useState(100);
  const [layers, setLayers] = useState<MapViewLayers>({
    basin: true,
    optical: false,
    sar: false,
    water: true,
    corridor: true,
    assets: true,
  });
  const sim = useSimulation();

  const runId = run?.run_id;
  const sarRevealed = sim.step === "obs-2" || sim.step === "obs-3";

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

  const beforeArea = 3.0; // baseline water area (km²) — from PRD demo script
  const afterArea = typeof run?.change_stats_json?.water_area_km2 === "number" ? run.change_stats_json.water_area_km2 : 4.1;
  const expansionPct = typeof run?.change_stats_json?.expansion_percent === "number" ? run.change_stats_json.expansion_percent : 14.3;

  // Initialize map once
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
      // Add basin polygon source (initialized with whatever we have at mount)
      map.addSource("basin", { type: "geojson", data: basinPolygon as any });
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

      // Change polygon (real water expansion from mask)
      map.addSource("water", {
        type: "geojson",
        data: (changePolygon ? { type: "Feature", geometry: (isPolygon(changePolygon) ? changePolygon : { type: "Polygon", coordinates: changePolygon.coordinates }), properties: {} } : { type: "Feature", geometry: { type: "Polygon", coordinates: [] }, properties: {} }) as any,
      });
      map.addLayer({
        id: "water-fill",
        type: "fill",
        source: "water",
        paint: { "fill-color": "#3B82F6", "fill-opacity": 0.4 * (opacity / 100) },
        layout: { visibility: layers.water ? "visible" : "none" },
      });
      map.addLayer({
        id: "water-outline",
        type: "line",
        source: "water",
        paint: { "line-color": "#3B82F6", "line-width": 1 },
        layout: { visibility: layers.water ? "visible" : "none" },
      });

      // Corridor source
      map.addSource("corridor", {
        type: "geojson",
        data: {
          type: "Feature",
          geometry: isLineString(corridorGeojson) ? corridorGeojson : (isFeatureCollection(corridorGeojson) ? { type: "LineString", coordinates: [] } : corridorGeojson),
          properties: {},
        } as any,
      });
      map.addLayer({
        id: "corridor-line",
        type: "line",
        source: "corridor",
        paint: { "line-color": "#F59E0B", "line-width": 2, "line-dasharray": [6, 4] },
        layout: { visibility: layers.corridor ? "visible" : "none" },
      });

      // Exposures / assets markers
      updateAssetMarkers(map, exposures, markersRef, sim.selectAsset, layers);

      // Raster overlays: baseline + current heatmap (image sources if bounds available)
      const bounds = mlEvidence.heatmap_bounds ?? mlEvidence.mask_bounds;
      if (bounds && bounds.length === 4) {
        map.addSource("heatmap", {
          type: "image",
          url: mlEvidence.heatmap_uri,
          coordinates: bounds as [[number, number], [number, number], [number, number], [number, number]],
        });
        map.addLayer({
          id: "heatmap-overlay",
          type: "raster",
          source: "heatmap",
          paint: { "raster-opacity": 0.7 * (opacity / 100), "raster-fade-duration": 0 },
          layout: { visibility: (layers.water || layers.sar) ? "visible" : "none" },
        });
      }
    });

    return () => {
      map.remove();
      mapRef.current = null;
      markersRef.current = [];
    };
  }, []);

  // Update map sources when data changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (map.getSource("basin") && basinPolygon) {
      (map.getSource("basin") as any).setData(basinPolygon);
    }
    if (map.getSource("water") && changePolygon) {
      const geom = isPolygon(changePolygon) ? changePolygon : { type: "Polygon", coordinates: changePolygon.coordinates ?? [] };
      (map.getSource("water") as any).setData({ type: "Feature", geometry: geom, properties: {} });
    }
    if (map.getSource("corridor") && corridorGeojson) {
      const geom = isLineString(corridorGeojson)
        ? corridorGeojson
        : (isFeatureCollection(corridorGeojson) && corridorGeojson.features[0]?.geometry) || { type: "LineString", coordinates: [] };
      (map.getSource("corridor") as any).setData({ type: "Feature", geometry: geom, properties: {} });
    }
    updateAssetMarkers(map, exposures, markersRef, sim.selectAsset, layers);

    // Update heatmap image source if present and bounds changed
    const bounds = mlEvidence.heatmap_bounds ?? mlEvidence.mask_bounds;
    if (bounds && bounds.length === 4 && map.getSource("heatmap")) {
      (map.getSource("heatmap") as any).updateImage({
        url: mlEvidence.heatmap_uri,
        coordinates: bounds as [[number, number], [number, number], [number, number], [number, number]],
      });
    }
  }, [basinPolygon, corridorGeojson, changePolygon, exposures, mlEvidence]);

  // Update opacity
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (map.getLayer("basin-fill")) {
      map.setPaintProperty("basin-fill", "fill-opacity", 0.08 * (opacity / 100));
    }
    if (map.getLayer("water-fill")) {
      map.setPaintProperty("water-fill", "fill-opacity", 0.4 * (opacity / 100));
    }
    if (map.getLayer("heatmap-overlay")) {
      map.setPaintProperty("heatmap-overlay", "raster-opacity", 0.7 * (opacity / 100));
    }
  }, [opacity]);

  // Fly to selected asset
  useEffect(() => {
    if (!sim.selectedAssetId || !mapRef.current) return;
    const asset = exposures.find((e) => e.asset_id === sim.selectedAssetId);
    if (!asset) return;
    // rough coordinate lookup by asset_id (exposures don't carry lat/lon yet)
    const coords = ASSET_COORDS[asset.asset_id];
    if (coords) {
      mapRef.current.flyTo({ center: coords, zoom: 14, duration: 1000 });
    }
  }, [sim.selectedAssetId, exposures]);

  const toggleLayer = (key: keyof MapViewLayers) => {
    setLayers((prev) => {
      const next = { ...prev, [key]: !prev[key] };
      const map = mapRef.current;
      if (map) {
        if (key === "basin") {
          map.setLayoutProperty("basin-fill", "visibility", next.basin ? "visible" : "none");
          map.setLayoutProperty("basin-border", "visibility", next.basin ? "visible" : "none");
        }
        if (key === "water") {
          map.setLayoutProperty("water-fill", "visibility", next.water ? "visible" : "none");
          map.setLayoutProperty("water-outline", "visibility", next.water ? "visible" : "none");
        }
        if (key === "corridor") map.setLayoutProperty("corridor-line", "visibility", next.corridor ? "visible" : "none");
        if (key === "sar" || key === "water") {
          if (map.getLayer("heatmap-overlay")) {
            map.setLayoutProperty("heatmap-overlay", "visibility", (next.water || next.sar) ? "visible" : "none");
          }
        }
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

  const beforeImg = mlEvidence.baseline_mask_uri;
  const afterImg = mlEvidence.heatmap_uri || mlEvidence.mask_uri;

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
            className="relative w-full h-[120px] rounded bg-surface-recessed overflow-hidden select-none cursor-ew-resize border border-border-subtle"
            onMouseMove={handleSwipe}
            onTouchMove={handleSwipe}
          >
            {/* After image (current observation heatmap/mask) */}
            <div className="absolute inset-0 bg-surface-canvas flex items-center justify-center">
              {afterImg ? (
                <img src={afterImg} alt="after" className="h-full w-full object-contain" />
              ) : (
                <span className="text-body-sm text-text-dim">After — no image</span>
              )}
            </div>
            {/* Before image (baseline), clipped by swipe position */}
            <div
              className="absolute inset-y-0 left-0 bg-surface-canvas overflow-hidden border-r-0 flex items-center justify-center"
              style={{ width: `${swipePct}%` }}
            >
              {beforeImg ? (
                <img src={beforeImg} alt="before" className="h-full w-[100vw] max-w-none object-contain" style={{ width: `${100 / (swipePct / 100 || 1)}%` }} />
              ) : (
                <span className="text-body-sm text-text-dim shrink-0">Before — no image</span>
              )}
            </div>
            {/* Labels */}
            <div className="absolute top-2 left-2 text-body-xs text-text-dim bg-surface-panel/80 px-1.5 rounded">
              Before — {beforeArea.toFixed(1)} km²
            </div>
            <div className="absolute top-2 right-2 text-body-xs text-primary-container bg-surface-panel/80 px-1.5 rounded">
              After — {afterArea.toFixed(1)} km² (+{expansionPct.toFixed(1)}%)
            </div>
            {/* Divider */}
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

          {sim.selectedAssetId && (
            <div className="pt-space-16 border-t border-border-subtle space-y-space-12">
              <AssetDetail assetId={sim.selectedAssetId} exposures={exposures} onJumpToReview={onJumpToReview} />
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}

// Fallback coordinates for demo assets (real OSM coordinates would come from backend in V2)
const ASSET_COORDS: Record<string, [number, number]> = {
  "village-2": [86.871, 27.904],
  "BR-12": [86.852, 27.913],
  "RD-4": [86.863, 27.907],
  "well-3": [86.861, 27.899],
};

function updateAssetMarkers(map: maplibregl.Map, exposures: any[], markersRef: React.MutableRefObject<maplibregl.Marker[]>, selectAsset: (id: string | null) => void, layers: MapViewLayers) {
  // clear old markers
  markersRef.current.forEach((m) => m.remove());
  markersRef.current = [];

  exposures.forEach((asset) => {
    const coords = asset.lon != null && asset.lat != null
      ? [asset.lon, asset.lat] as [number, number]
      : ASSET_COORDS[asset.asset_id];
    if (!coords) return;
    const status = asset.inundated ? "red" : (asset.distance_m != null ? "amber" : "green");
    const el = document.createElement("div");
    el.style.cssText = `width: 12px; height: 12px; border-radius: 50%; background: ${STATUS_COLOR[status]}; border: 2px solid #0A0F1E; cursor: pointer;`;
    el.title = asset.name || asset.asset_id;
    el.onclick = () => selectAsset(asset.asset_id);
    const marker = new maplibregl.Marker(el).setLngLat(coords).addTo(map);
    markersRef.current.push(marker);
  });
}

function AssetDetail({ assetId, exposures, onJumpToReview }: { assetId: string; exposures: any[]; onJumpToReview?: () => void }) {
  const asset = exposures.find((e) => e.asset_id === assetId);
  if (!asset) return null;
  const status = asset.inundated ? "red" : asset.distance_m != null ? "amber" : "green";
  return (
    <div className="space-y-space-12">
      <div className="space-y-space-4">
        <span className="text-caption text-text-dim uppercase tracking-wider">Target Inspection</span>
        <h3 className="text-headline-md font-medium text-text-primary">{asset.name}</h3>
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
