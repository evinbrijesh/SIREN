import { useEffect, useState } from "react";

export function useOnlineStatus(): boolean {
  const [online, setOnline] = useState(() => typeof navigator === "undefined" || navigator.onLine);

  useEffect(() => {
    const handleOnline = () => setOnline(true);
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
  return (
    <span className={`data-val text-body-sm ${online ? "text-text-dim" : "text-status-safe"}`} aria-live="polite">
      {online ? "ONLINE" : "OFFLINE — ALL SYSTEMS LOCAL"}
    </span>
  );
}
