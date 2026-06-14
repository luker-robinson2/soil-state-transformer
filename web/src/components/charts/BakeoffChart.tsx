import { useMemo, useState } from 'react';
import { motion } from 'motion/react';
import { BAKEOFF, TARGETS, type Target } from '../../data/results';
import { fmt3, scale, easeOut } from './util';

/*
  Bake-off: R² against OpenLandMap under spatial-block CV. Pick a property;
  models re-sort and bars re-animate. GeoSoil (end-to-end fusion) and the
  frozen-latent probes are colour-keyed apart from the tabular baselines so
  the reviewer can see the representation is competitive, not just the model.
*/

const X0 = 150;
const X1 = 560;
const ROW_H = 34;
const TOP = 10;

const KIND_COLOR: Record<string, string> = {
  tabular: 'var(--color-line-strong)',
  'frozen-probe': 'var(--color-moss-300)',
  geosoil: 'var(--color-moss-600)',
};
const KIND_LABEL: Record<string, string> = {
  tabular: 'Tabular baseline',
  'frozen-probe': 'Frozen latent (probe)',
  geosoil: 'GeoSoil (end-to-end)',
};

export default function BakeoffChart() {
  const [target, setTarget] = useState<Target>('soc');

  const rows = useMemo(() => {
    return [...BAKEOFF].sort((a, b) => a.r2[target] - b.r2[target]);
  }, [target]);

  // domain: a little headroom below the min, up to ~0.95
  const vals = rows.map((r) => r.r2[target]);
  const lo = Math.max(0, Math.floor((Math.min(...vals) - 0.06) * 20) / 20);
  const hi = Math.min(1, Math.ceil((Math.max(...vals) + 0.03) * 20) / 20);
  const height = TOP * 2 + rows.length * ROW_H + 18;

  const ticks = useMemo(() => {
    const out: number[] = [];
    for (let v = lo; v <= hi + 1e-9; v += 0.1) out.push(Math.round(v * 100) / 100);
    return out;
  }, [lo, hi]);

  return (
    <div className="not-prose w-full">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1.5">
          {TARGETS.map((t) => {
            const on = t.key === target;
            return (
              <button
                key={t.key}
                onClick={() => setTarget(t.key as Target)}
                className={[
                  'rounded-lg px-3 py-1.5 text-sm font-medium transition-all duration-200',
                  on
                    ? 'bg-moss-700 text-paper'
                    : 'bg-paper-2 text-ink-2 hover:bg-moss-100',
                ].join(' ')}
              >
                {t.label}
                <span className="ml-1.5 font-mono text-[0.7rem] opacity-60">
                  {t.unit || '—'}
                </span>
              </button>
            );
          })}
        </div>
        <div className="flex gap-3 font-mono text-[0.68rem] text-ink-3">
          {(['tabular', 'frozen-probe', 'geosoil'] as const).map((k) => (
            <span key={k} className="flex items-center gap-1.5">
              <span
                className="h-2.5 w-2.5 rounded-[3px]"
                style={{ background: KIND_COLOR[k] }}
              />
              {KIND_LABEL[k]}
            </span>
          ))}
        </div>
      </div>

      <div className="rounded-2xl border border-line bg-card p-3 sm:p-5">
        <svg viewBox={`0 0 580 ${height}`} className="w-full" role="img"
          aria-label={`Bake-off R-squared for ${target}`}>
          {ticks.map((g) => {
            const x = scale(g, lo, hi, X0, X1);
            return (
              <g key={g}>
                <line x1={x} x2={x} y1={TOP} y2={TOP + rows.length * ROW_H}
                  stroke="var(--color-line)" strokeWidth={1} strokeDasharray="2 4" />
                <text x={x} y={height - 2} textAnchor="middle" className="fill-ink-faint"
                  style={{ fontFamily: 'var(--font-mono)', fontSize: 10 }}>
                  {g.toFixed(1)}
                </text>
              </g>
            );
          })}

          {rows.map((r, i) => {
            const v = r.r2[target];
            const y = TOP + i * ROW_H;
            const barH = ROW_H - 13;
            const w = scale(v, lo, hi, X0, X1) - X0;
            const isGeo = r.kind === 'geosoil';
            return (
              <g key={r.key}>
                <motion.text
                  x={X0 - 12} className={isGeo ? 'fill-moss-700' : 'fill-ink-2'}
                  style={{ fontFamily: 'var(--font-sans)', fontSize: 12.5,
                    fontWeight: isGeo ? 700 : 500 }}
                  textAnchor="end"
                  initial={false} animate={{ y: y + barH / 2 + 4 }}
                  transition={{ duration: 0.6, ease: easeOut }}
                >
                  {r.name}
                </motion.text>
                <motion.rect
                  x={X0} height={barH} rx={4} fill={KIND_COLOR[r.kind]}
                  initial={false} animate={{ y: y + 2, width: Math.max(0, w) }}
                  transition={{ duration: 0.7, ease: easeOut }}
                />
                <motion.text
                  className={isGeo ? 'fill-paper' : 'fill-ink'}
                  style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5,
                    fontWeight: 600 }}
                  initial={false}
                  animate={{
                    x: isGeo ? X0 + w - 8 : X0 + w + 8,
                    y: y + barH / 2 + 4,
                  }}
                  textAnchor={isGeo ? 'end' : 'start'}
                  transition={{ duration: 0.7, ease: easeOut }}
                >
                  {fmt3(v)}
                </motion.text>
              </g>
            );
          })}
        </svg>
      </div>
      <p className="mt-3 font-mono text-[0.68rem] text-ink-faint">
        R² vs OpenLandMap · 1°×1° spatial-block GroupKFold · n=3000 ·
        sources: baselines.json, cv_transformer.json, hybrid.json
      </p>
    </div>
  );
}
