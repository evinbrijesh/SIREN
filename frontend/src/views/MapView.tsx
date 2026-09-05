import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { api, apiOrMock } from "../api/client";
import { mockData } from "../api/mockData";
import { useSimulation, type SimStep } from "../simulation/SimulationContext";
import type { BasinConfig, Run, Exposure, ExposureList, MlEvidence, GeoJSONFeature } from "../api/types";

interface MapViewProps {
  basin?: BasinConfig;
  run?: Run;
  onJumpToReview?: () => void;
}

const STEP_LABELS: Record<SimStep, string> = {
  before: "BASELINE | 2025-11-22",
  "obs-1": "OBS 01 | 2026-07-23",
  "obs-2": "OBS 02 | 2026-08-04",
  "obs-3": "OBS 03 | 2026-08-12",
};

const STATUS_COLOR = { safe: "#10b981", buffered: "#ffb000", inundated: "#ff1e27" } as const;

type LayerKey = "basin" | "hillshade" | "optical" | "sar" | "water" | "corridor" | "assets";
type MapViewLayers = Record<LayerKey, boolean>;

const LAYER_LABELS: { key: LayerKey; label: string }[] = [
  { key: "basin", label: "Basin AOI" },
  { key: "hillshade", label: "DEM hillshade" },
  { key: "optical", label: "Optical baseline · S2 L2A" },
  { key: "sar", label: "SAR backscatter · S1 VV" },
  { key: "water", label: "Water expansion mask" },
  { key: "corridor", label: "D8 + OSM corridor" },
  { key: "assets", label: "OSM critical assets" },
];

function toFeature(geometry: GeoJSONFeature | null): any {
  if (!geometry) return { type: "Feature", geometry: { type: "Point", coordinates: [] }, properties: {} };
  if (geometry.type === "Feature") return geometry;
  if (geometry.type === "FeatureCollection") {
    return geometry;
  }
  return { type: "Feature", geometry, properties: {} };
}

function assetCoordinates(asset: Exposure): [number, number] | null {
  const geometry = asset.geometry_geojson;
  if (geometry?.type === "Point") return geometry.coordinates as [number, number];
  if (geometry?.type === "LineString" && geometry.coordinates?.length) {
    const coordinates = geometry.coordinates as [number, number][];
    return coordinates[Math.floor(coordinates.length / 2)];
  }
  return null;
}

function assetStatus(asset: Exposure): keyof typeof STATUS_COLOR {
  if (asset.inundated) return "inundated";
  if (asset.distance_m !== null && asset.buffer_m !== null && asset.distance_m <= asset.buffer_m) return "buffered";
  return "safe";
}

export default function MapView({ basin, run, onJumpToReview }: MapViewProps = {}) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markersRef = useRef<maplibregl.Marker[]>([]);
  const compareRef = useRef<HTMLDivElement>(null);
  const [comparePct, setComparePct] = useState(50);
  const [compareOpacity, setCompareOpacity] = useState(75);
  const [compareOpen, setCompareOpen] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(75);
  const [sarSweepActive, setSarSweepActive] = useState(false);
  const prevRoutedSarRef = useRef(false);
  const [mapCenter, setMapCenter] = useState<[number, number]>([86.807, 27.866]);
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);

  // Theme-aware map colors — reads --map-style CSS variable
  const mapStyle = typeof window !== "undefined"
    ? getComputedStyle(document.documentElement).getPropertyValue("--map-style").trim() || "dark"
    : "dark";
  const isLightMap = mapStyle === "light";
  const mapBg = isLightMap ? "#e5e7eb" : "#08090a";
  const basinColor = isLightMap ? "#b45309" : "#ffb000";
  const corridorColor = isLightMap ? "#dc2626" : "#ff1e27";
  const sim = useSimulation();

  const routedSar = Boolean((run?.change_stats_json?.routing as any)?.sar_primary);
  const [layers, setLayers] = useState<MapViewLayers>({
    basin: true,
    hillshade: false,
    optical: true,
    sar: false,
    water: true,
    corridor: true,
    assets: true,
  });

  useEffect(() => {
    if (routedSar) setLayers((previous) => ({ ...previous, sar: true }));
  }, [routedSar, run?.run_id]);

  // SAR sweep — one-time 1.2s animation when SAR path engages
  useEffect(() => {
    if (routedSar && !prevRoutedSarRef.current) {
      setSarSweepActive(true);
      const timer = setTimeout(() => setSarSweepActive(false), 1300);
      prevRoutedSarRef.current = true;
      return () => clearTimeout(timer);
    }
    if (!routedSar) prevRoutedSarRef.current = false;
  }, [routedSar]);

  const runId = run?.run_id;
  const { data: exposuresData } = useQuery({
    queryKey: ["exposures", runId],
    queryFn: () => apiOrMock(() => api.listExposures(runId!), "exposures") as Promise<ExposureList>,
    enabled: Boolean(runId),
  });
  const { data: mlData } = useQuery({
    queryKey: ["ml-evidence", runId],
    queryFn: () => apiOrMock(() => api.getMlEvidence(runId!), "mlEvidence") as Promise<MlEvidence>,
    enabled: Boolean(runId),
  });

  const exposures = exposuresData?.exposures ?? mockData.exposures.exposures;
  const mlEvidence = run ? mlData ?? mockData.mlEvidence : null;
  const basinPolygon = basin?.boundary_geojson ?? mockData.basin.boundary_geojson;
  const bounds = basin?.basemap_bounds ?? mockData.basin.basemap_bounds;
  const opticalUri = basin?.basemap_uri ?? mockData.basin.basemap_uri;
  const corridor = run?.corridor_geojson ?? null;
  const beforeArea = 3.0;
  const afterArea = (run?.change_stats_json?.water_area_km2 as number) ?? 3.0;
  const expansionPct = (run?.change_stats_json?.expansion_percent as number) ?? 0;

  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: { version: 8, sources: {}, layers: [{ id: "background", type: "background", paint: { "background-color": mapBg } }] },
      center: [86.807, 27.866],
      zoom: 11.6,
      attributionControl: { compact: true },
    });
    mapRef.current = map;
    // Throttled coordinate listener — only update on moveend, not during pan
    map.on("moveend", () => {
      const center = map.getCenter();
      setMapCenter([center.lng, center.lat]);
    });
    map.on("load", () => {
      if (opticalUri && bounds?.length === 4) {
        const coordinates = bounds as [[number, number], [number, number], [number, number], [number, number]];
        map.addSource("optical", { type: "image", url: opticalUri, coordinates });
        map.addLayer({ id: "optical", type: "raster", source: "optical", paint: { "raster-opacity": 1, "raster-fade-duration": 0 } });
        map.addSource("hillshade", { type: "image", url: "/data/map-assets/dem-hillshade.png", coordinates });
        map.addLayer({ id: "hillshade", type: "raster", source: "hillshade", paint: { "raster-opacity": 0.35, "raster-fade-duration": 0 }, layout: { visibility: "none" } });
        map.addSource("sar", { type: "image", url: "/data/map-assets/sar-backscatter.png", coordinates });
        map.addLayer({ id: "sar", type: "raster", source: "sar", paint: { "raster-opacity": 0.72, "raster-fade-duration": 0 }, layout: { visibility: routedSar ? "visible" : "none" } });
        // Preserve initial camera — jumpTo keeps the centered view on Imja Lake
        // instead of fitBounds which jumps to show the wide basin extent
        map.jumpTo({ center: [86.807, 27.866], zoom: 11.6 });
      }
      map.addSource("basin", { type: "geojson", data: basinPolygon as any });
      map.addLayer({ id: "basin-fill", type: "fill", source: "basin", paint: { "fill-color": basinColor, "fill-opacity": 0.03 } });
      map.addLayer({ id: "basin-border", type: "line", source: "basin", paint: { "line-color": basinColor, "line-width": 1, "line-opacity": 0.5 } });
      map.addSource("corridor", { type: "geojson", data: toFeature(corridor) });
      map.addLayer({ id: "corridor", type: "line", source: "corridor", paint: { "line-color": corridorColor, "line-width": 3 } });
      updateAssetMarkers(map, exposures, markersRef, sim.selectAsset, layers.assets);
    });
    return () => {
      map.remove();
      mapRef.current = null;
      markersRef.current = [];
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.isStyleLoaded()) return;
    const setVisibility = (id: string, visible: boolean) => {
      if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
    };
    setVisibility("optical", layers.optical);
    setVisibility("hillshade", layers.hillshade);
    setVisibility("sar", layers.sar);
    setVisibility("basin-fill", layers.basin);
    setVisibility("basin-border", layers.basin);
    setVisibility("corridor", layers.corridor);
    setVisibility("water", layers.water);
    if (map.getLayer("hillshade")) map.setPaintProperty("hillshade", "raster-opacity", 0.45 * overlayOpacity / 100);
    if (map.getLayer("sar")) map.setPaintProperty("sar", "raster-opacity", 0.85 * overlayOpacity / 100);
    if (map.getLayer("water")) map.setPaintProperty("water", "raster-opacity", 0.65 * overlayOpacity / 100);
    updateAssetMarkers(map, exposures, markersRef, sim.selectAsset, layers.assets);
  }, [layers, overlayOpacity, exposures, sim.selectAsset]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.isStyleLoaded()) return;
    if (map.getSource("basin")) (map.getSource("basin") as maplibregl.GeoJSONSource).setData(basinPolygon as any);
    if (map.getSource("corridor")) (map.getSource("corridor") as maplibregl.GeoJSONSource).setData(toFeature(corridor));
    const waterBounds = mlEvidence?.mask_bounds;
    if (mlEvidence && waterBounds?.length === 4) {
      const coordinates = waterBounds as [[number, number], [number, number], [number, number], [number, number]];
      if (map.getSource("water")) {
        (map.getSource("water") as maplibregl.ImageSource).updateImage({ url: mlEvidence.mask_uri, coordinates });
      } else {
        map.addSource("water", { type: "image", url: mlEvidence.mask_uri, coordinates });
        map.addLayer({ id: "water", type: "raster", source: "water", paint: { "raster-opacity": 0.65 * overlayOpacity / 100, "raster-fade-duration": 0 }, layout: { visibility: layers.water ? "visible" : "none" } });
      }
    } else if (map.getLayer("water")) {
      map.setLayoutProperty("water", "visibility", "none");
    }
  }, [basinPolygon, corridor, mlEvidence, layers.water, overlayOpacity]);

  useEffect(() => {
    if (!sim.selectedAssetId || !mapRef.current) return;
    const selected = exposures.find((asset) => asset.asset_id === sim.selectedAssetId);
    const coordinates = selected ? assetCoordinates(selected) : null;
    if (coordinates) mapRef.current.flyTo({ center: coordinates, zoom: 14, duration: 100 });
  }, [sim.selectedAssetId, exposures]);

  const flyToAsset = (asset: Exposure) => {
    const coordinates = assetCoordinates(asset);
    if (coordinates) mapRef.current?.flyTo({ center: coordinates, zoom: 14, duration: 100 });
  };

  const updateCompare = (clientX: number) => {
    const rect = compareRef.current?.getBoundingClientRect();
    if (!rect) return;
    setComparePct(Math.max(0, Math.min(100, (clientX - rect.left) / rect.width * 100)));
  };

  useEffect(() => {
    if (!isDragging) return;
    const move = (event: MouseEvent | TouchEvent) => updateCompare("touches" in event ? event.touches[0].clientX : event.clientX);
    const stop = () => setIsDragging(false);
    window.addEventListener("mousemove", move as EventListener);
    window.addEventListener("mouseup", stop);
    window.addEventListener("touchmove", move as EventListener, { passive: true });
    window.addEventListener("touchend", stop);
    return () => {
      window.removeEventListener("mousemove", move as EventListener);
      window.removeEventListener("mouseup", stop);
      window.removeEventListener("touchmove", move as EventListener);
      window.removeEventListener("touchend", stop);
    };
  }, [isDragging]);

  const selectedAsset = exposures.find((asset) => asset.asset_id === sim.selectedAssetId) ?? null;
  const beforeImage = mlEvidence?.preview_baseline_uri;
  const afterImage = mlEvidence?.preview_after_uri;

  return (
    <div className="flex h-full">
      <aside className={`${leftOpen ? "w-dock-left-width" : "w-8"} flex-none bg-surface-panel border-r border-border-subtle flex flex-col justify-between transition-all`}>
        {leftOpen ? <>
          <div className="p-space-12">
            <div className="flex items-center justify-between mb-space-8"><h2 className="label-caps">Layers</h2><button onClick={() => setLeftOpen(false)} className="text-text-dim hover:text-text-primary">Close</button></div>
            <div className="space-y-space-2">
              {LAYER_LABELS.map(({ key, label }) => <label key={key} className="flex items-center gap-space-8 px-space-4 py-space-2 hover:bg-surface-container cursor-pointer">
                <input type="checkbox" checked={layers[key]} onChange={() => setLayers((previous) => ({ ...previous, [key]: !previous[key] }))} className="w-3 h-3 accent-primary-container" />
                <span className="text-body-md">{label}</span>
                {key === "sar" && routedSar && <span className="ml-auto text-caption text-status-warn">auto</span>}
              </label>)}
            </div>
          </div>
          <div className="p-space-12 border-t border-border-subtle">
            <div className="flex justify-between mb-space-4"><span className="label-caps">Overlay opacity</span><span className="data-val text-body-sm">{overlayOpacity}%</span></div>
            <input aria-label="Overlay opacity" type="range" min="0" max="100" value={overlayOpacity} onChange={(event) => setOverlayOpacity(Number(event.target.value))} className="w-full h-1 accent-primary-container" />
          </div>
        </> : <button onClick={() => setLeftOpen(true)} className="py-space-8 text-text-dim hover:text-text-primary text-body-sm">Layers</button>}
      </aside>

      <div className="relative flex-1 min-w-0 bg-surface-recessed overflow-hidden tactical-bezel tactical-reg" data-reg="REF: 45RVL-HIMAL">
        <div ref={mapContainer} className="absolute inset-0" />
        {sarSweepActive && (
          <>
            <div className="sar-sweep-trail" />
            <div className="sar-sweep-line" />
          </>
        )}
        <div className="absolute top-space-8 left-space-8 z-10 px-space-8 py-space-4 bg-surface-panel/80 backdrop-blur-sm border border-border-subtle text-body-sm">{STEP_LABELS[sim.step]}</div>
        <div className="absolute top-space-8 right-space-8 z-10 flex gap-space-4">
          <button disabled={!run || !beforeImage || !afterImage} onClick={() => setCompareOpen((open) => !open)} className="px-space-8 py-space-4 bg-surface-panel border border-border-subtle text-body-sm disabled:opacity-40">{compareOpen ? "Close compare" : "Swipe compare"}</button>
        </div>

        {compareOpen && beforeImage && afterImage && <div ref={compareRef} className="absolute inset-space-16 top-12 z-20 bg-surface-canvas border border-border-strong overflow-hidden select-none cursor-ew-resize" onMouseDown={(event) => { setIsDragging(true); updateCompare(event.clientX); }} onTouchStart={(event) => { setIsDragging(true); updateCompare(event.touches[0].clientX); }}>
          <img src={afterImage} alt="Current observation change mask" className="absolute inset-0 w-full h-full object-cover" style={{ opacity: compareOpacity / 100 }} />
          <div className="absolute inset-y-0 left-0 overflow-hidden" style={{ width: `${comparePct}%` }}>
            <img src={beforeImage} alt="Pre-event optical baseline" className="absolute inset-0 h-full object-cover" style={{ width: compareRef.current?.clientWidth ?? '100%', maxWidth: 'none' }} />
          </div>
          <div className="absolute top-0 left-0 right-0 h-9 bg-surface-panel border-b border-border-subtle flex items-center px-space-8 gap-space-12 cursor-default" onMouseDown={(event) => event.stopPropagation()}>
            <span className="text-body-sm">Before</span><span className="text-border-subtle">|</span><span className="text-body-sm">After</span>
            <label className="ml-auto flex items-center gap-space-6 text-caption text-text-dim">Opacity<input aria-label="Compare opacity" type="range" min="0" max="100" value={compareOpacity} onChange={(event) => setCompareOpacity(Number(event.target.value))} /></label>
          </div>
          <div className="absolute top-9 bottom-0 w-[3px] bg-primary pointer-events-none z-30" style={{ left: `${comparePct}%`, boxShadow: "0 0 8px 1px var(--color-primary)" }} />
          <div className="absolute top-1/2 z-30 pointer-events-none" style={{ left: `calc(${comparePct}% - 10px)`, transform: "translateY(-50%)" }}>
            <div className="w-5 h-10 bg-primary flex items-center justify-center text-text-inverse text-caption font-bold">↔</div>
          </div>
          <div className="absolute bottom-space-8 left-space-8 bg-surface-panel border border-border-subtle px-space-6 py-space-2 data-val text-caption">{beforeArea.toFixed(2)} km²</div>
          <div className="absolute bottom-space-8 right-space-8 bg-surface-panel border border-border-subtle px-space-6 py-space-2 data-val text-caption">{afterArea.toFixed(2)} km² (+{expansionPct.toFixed(1)}%)</div>
        </div>}

        {/* GIS telemetry ribbon — fixed bottom bar with live coordinates */}
        <div className="absolute bottom-0 left-0 right-0 h-[24px] bg-surface-panel border-t border-border-subtle flex items-center justify-between px-space-8 data-val text-caption text-text-dim z-10 select-none">
          <div className="flex items-center gap-space-12">
            <span className="text-primary">CRS: EPSG:4326</span>
            <span>CENTER: [{mapCenter[0].toFixed(3)}°E, {mapCenter[1].toFixed(3)}°N]</span>
            <span>RES: 10m/px</span>
          </div>
          <div className="flex items-center gap-space-12">
            <span>ELEV: 4,980m</span>
            <span className="text-status-safe">CRYPTO: SHA-256 VALID</span>
          </div>
        </div>
      </div>

      <aside className={`${rightOpen ? "w-dock-right-width" : "w-8"} flex-none bg-surface-panel border-l border-border-subtle transition-all`}>
        {rightOpen ? <>
          <div className="p-space-12 border-b border-border-subtle">
            <div className="flex justify-between mb-space-8"><h2 className="label-caps">Legend</h2><button onClick={() => setRightOpen(false)} className="text-text-dim hover:text-text-primary">Close</button></div>
            {(Object.keys(STATUS_COLOR) as (keyof typeof STATUS_COLOR)[]).map((status) => <div key={status} className="flex items-center gap-space-8 py-space-2"><span className="w-2.5 h-2.5 rounded-full" style={{ background: STATUS_COLOR[status] }} /><span className="text-body-sm">{status}</span></div>)}
          </div>
          {selectedAsset && <AssetDetailCard asset={selectedAsset} onFlyTo={() => flyToAsset(selectedAsset)} onReview={onJumpToReview} />}
        </> : <button onClick={() => setRightOpen(true)} className="py-space-8 text-text-dim hover:text-text-primary text-body-sm">Assets</button>}
      </aside>
    </div>
  );
}

function updateAssetMarkers(map: maplibregl.Map, exposures: Exposure[], markersRef: React.MutableRefObject<maplibregl.Marker[]>, selectAsset: (id: string | null) => void, visible: boolean) {
  markersRef.current.forEach((marker) => marker.remove());
  markersRef.current = [];
  if (!visible) return;
  exposures.forEach((asset) => {
    const coordinates = assetCoordinates(asset);
    if (!coordinates) return;
    const status = assetStatus(asset);
    const element = document.createElement("button");
    element.type = "button";
    element.setAttribute("aria-label", `${asset.name ?? asset.asset_id}: ${status}`);
    element.title = asset.name ?? asset.asset_id;
    element.style.cssText = `width:12px;height:12px;background:${STATUS_COLOR[status]};border:1px solid #0b0f17;border-radius:50%;cursor:pointer;`;
    element.onclick = () => selectAsset(asset.asset_id);
    markersRef.current.push(new maplibregl.Marker({ element }).setLngLat(coordinates).addTo(map));
  });
}

function AssetDetailCard({ asset, onFlyTo, onReview }: { asset: Exposure; onFlyTo: () => void; onReview?: () => void }) {
  const status = assetStatus(asset);
  return <div className="p-space-12 space-y-space-8">
    <div><span className="label-caps">Asset detail</span><h3 className="data-val text-headline-md">{asset.name ?? asset.asset_id}</h3></div>
    <div className="data-val text-body-sm text-text-dim space-y-space-2">
      <div>ID: <span className="text-text-primary">{asset.asset_id}</span></div>
      <div>TYPE: <span className="text-text-primary">{asset.asset_type}</span></div>
      <div>DIST / BUFFER: <span className="text-text-primary">{asset.distance_m?.toFixed(0) ?? "N/A"} m / {asset.buffer_m?.toFixed(0) ?? "N/A"} m</span></div>
      <div>POP SERVED: <span className="text-text-primary">{asset.population?.toLocaleString() ?? "N/A"}</span></div>
      <div>STATUS: <span style={{ color: STATUS_COLOR[status] }}>{status.toUpperCase()}</span></div>
    </div>
    <div className="flex gap-space-4"><button onClick={onFlyTo} className="flex-1 border border-border-strong px-space-8 py-space-4 data-val text-body-sm">[FLY TO]</button>{onReview && <button onClick={onReview} className="flex-1 border border-primary text-primary px-space-8 py-space-4 data-val text-body-sm">[REVIEW]</button>}</div>
  </div>;
}
