import { useTheme, type ThemeName } from "./ThemeContext";

const LABELS: Record<ThemeName, string> = {
  "ops-dark": "Ops Dark",
  "professional-light": "Light",
  satellite: "Satellite",
};

const ICONS: Record<ThemeName, string> = {
  "ops-dark": "🌙",
  "professional-light": "☀️",
  satellite: "🛰️",
};

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  return (
    <div className="flex items-center gap-1 rounded bg-surface-panel border border-border-subtle p-1">
      {(Object.keys(LABELS) as ThemeName[]).map((t) => (
        <button
          key={t}
          onClick={() => setTheme(t)}
          title={LABELS[t]}
          className={`px-2 py-1 text-xs rounded transition-colors ${
            theme === t
              ? "bg-primary text-primary-fg font-medium"
              : "text-text-dim hover:text-text-primary hover:bg-surface-recessed"
          }`}
        >
          <span className="mr-1">{ICONS[t]}</span>
          {LABELS[t]}
        </button>
      ))}
    </div>
  );
}
