import { useTheme, type ThemeName } from "./ThemeContext";

const LABELS: Record<ThemeName, string> = {
  "ops-dark": "OPS",
  "professional-light": "LIGHT",
  satellite: "SAT",
};

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  return (
    <div className="flex items-stretch border border-border-subtle">
      {(Object.keys(LABELS) as ThemeName[]).map((t) => (
        <button
          key={t}
          onClick={() => setTheme(t)}
          title={LABELS[t]}
          className={`px-space-8 py-space-2 text-body-sm data-val transition-colors ${
            theme === t
              ? "bg-surface-container text-text-primary"
              : "text-text-dim hover:text-text-primary hover:bg-surface-recessed"
          }`}
        >
          {LABELS[t]}
        </button>
      ))}
    </div>
  );
}
