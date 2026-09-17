import { describe, expect, it, vi, afterEach } from "vitest";
import { sendNtfyAlert } from "./ntfy";

function setOnline(online: boolean) {
  Object.defineProperty(navigator, "onLine", {
    value: online,
    configurable: true,
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  setOnline(true);
});

describe("sendNtfyAlert", () => {
  it("returns simulated dispatch without fetch when offline (air-gap)", () => {
    setOnline(false);
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const onResult = vi.fn();

    sendNtfyAlert({ expansionPct: 42 }, onResult);

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(onResult).toHaveBeenCalledWith(
      false,
      expect.stringContaining("Air-gap mode"),
    );
  });

  it("reports success when fetch resolves online", async () => {
    setOnline(true);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
    const onResult = vi.fn();

    sendNtfyAlert({ expansionPct: 10 }, onResult);
    await vi.waitFor(() => expect(onResult).toHaveBeenCalled());

    expect(onResult).toHaveBeenCalledWith(true, expect.stringContaining("Live alert"));
  });

  it("falls back to simulated dispatch when fetch rejects", async () => {
    setOnline(true);
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("no network")));
    const onResult = vi.fn();

    sendNtfyAlert({}, onResult);
    await vi.waitFor(() => expect(onResult).toHaveBeenCalled());

    expect(onResult).toHaveBeenCalledWith(
      false,
      expect.stringContaining("simulated dispatch"),
    );
  });

  it("includes alert details in the body when decoded payload present", async () => {
    setOnline(true);
    const fetchSpy = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", fetchSpy);

    sendNtfyAlert(
      {
        expansionPct: 33,
        decoded: {
          alert_id: "siren-0001",
          sector: "sector-b",
          hazard: "0.88",
          severity: "critical",
          exposed_pop: 412,
          critical_assets: ["Well 3"],
          medical_action: "boil water",
        },
      },
    );

    await vi.waitFor(() => expect(fetchSpy).toHaveBeenCalled());
    const body = fetchSpy.mock.calls[0][1].body as string;
    expect(body).toContain("siren-0001");
    expect(body).toContain("sector-b");
    expect(body).toContain("+33%");
    expect(body).toContain("BOIL WATER");
  });
});
