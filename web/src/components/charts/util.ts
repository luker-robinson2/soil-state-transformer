export const fmt2 = (x: number) => (x >= 0 ? x.toFixed(2) : `−${Math.abs(x).toFixed(2)}`);
export const fmt3 = (x: number) => (x >= 0 ? x.toFixed(3) : `−${Math.abs(x).toFixed(3)}`);

// Map an R² value to a horizontal-axis x position within [x0, x1] for a domain [lo, hi].
export const scale = (v: number, lo: number, hi: number, x0: number, x1: number) =>
  x0 + ((v - lo) / (hi - lo)) * (x1 - x0);

export const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

export const easeOut = [0.22, 1, 0.36, 1] as const;
