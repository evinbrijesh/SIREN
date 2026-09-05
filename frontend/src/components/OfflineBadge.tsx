import { useEffect, useState } from "react";
import { getLastStaleness, getOutboxCount, flushOutbox } from "../api/client";

export function useOnlineStatus(): boolean {
  const [online, setOnline] = useState(() => typeof navigator === "undefined" || navigator.onLine);

  useEffect(() => {
    const handleOnline = () => {
      setOnline(true);
      // Auto-flush outbox when connection is restored
      flushOutbox().then(() => {});
    };
    const handleOffline = () => setOnline(false);
    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  return online;
}

export default function OfflineBadge() {
  const online = useOnlineStatus();
  const [outboxCount, setOutboxCount] = useState(0);
  const staleness = getLastStaleness();

  useEffect(() => {
    setOutboxCount(getOutboxCount());
    const interval = setInterval(() => setOutboxCount(getOutboxCount()), 2000);
    return () => clearInterval(interval);
  }, [online]);

  const stale = staleness.isStale && !online;

  return (
    <div className="flex items-center gap-space-6" aria-live="polite">
      <span className={`data-val text-body-sm ${online ? "text-text-dim" : "text-status-safe"}`}>
        {online ? "ONLINE" : "OFFLINE — ALL SYSTEMS LOCAL"}
      </span>
      {stale && (
        <span className="data-val text-caption text-status-warn border border-status-warn px-space-4 py-space-1">
          STALE {Math.floor(staleness.ageMs / 1000)}s
        </span>
      )}
      {outboxCount > 0 && (
        <span className="data-val text-caption text-status-warn border border-status-warn px-space-4 py-space-1">
          OUTBOX: {outboxCount} QUEUED
        </span>
      )}
    </div>
  );
}
