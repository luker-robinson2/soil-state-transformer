import { useState } from 'react';
import { motion } from 'motion/react';
import { MULTI_TRUTH, TARGETS, type Target } from '../../data/results';
import { fmt2, scale, easeOut } from './util';

/*
  The standout exhibit. Same model family, three independent ground truths.
  Switching the truth re-animates the bars so the reviewer *sees* texture R²
  collapse below zero against lab measurements — then partially recover when
  radar + bare-soil modalities are added. This is the multi-truth finding.
*/

const LO = -0.3;
const HI = 1.0;
// plotting box (viewBox units)
const X0 = 92;
const X1 = 540;
const ROW_H = 52;
const TOP = 16;

const NARRATIVE: Record<string, string> = {
  openlandmap:
    'Against OpenLandMap — a modelled label product derived from the same satellite covariates — every property scores well. This is the number a single-leaderboard paper would report.',
  'kssl-ae':
    'Trained and tested directly on lab-measured pedons, AlphaEarth alone predicts texture and bulk density at R² ≤ 0. The apparent texture skill above was largely circular: model and label shared inputs.',
  'kssl-mm':
    'Adding radar (Sentinel-1), bare-soil composites and terrain recovers real, lab-verifiable signal — sand to R² ≈ 0.34, clay ≈ 0.25. Honest, modest, and not visible without multiple ground truths.',
};

export default function MultiTruthPanel() {
  const [active, setActive] = useState(0);
  const series = MULTI_TRUTH[active];
  const zeroX = scale(0, LO, HI, X0, X1);
  const height = TOP * 2 + TARGETS.length * ROW_H;

  return (
    <div className="not-prose w-full">
      {/* truth selector */}
      <div className="mb-5 flex flex-wrap gap-2">
        {MULTI_TRUTH.map((s, i) => {
          const on = i === active;
          return (
            <button
              key={s.key}
              onClick={() => setActive(i)}
              className={[
                'group relative rounded-xl border px-3.5 py-2 text-left transition-all duration-300',
                on
                  ? 'border-moss-400 bg-card shadow-[0_1px_0_rgba(0,0,0,0.02)]'
                  : 'border-line bg-transparent hover:border-line-strong',
              ].join(' ')}
            >
              <div className="flex items-center gap-2">
                <span
                  className="h-2 w-2 rounded-full transition-colors"
                  style={{
                    background:
                      s.tone === 'good'
                        ? 'var(--color-moss-500)'
                        : s.tone === 'warn'
                          ? 'var(--color-moss-300)'
                          : 'var(--color-clay)',
                  }}
                />
                <span
                  className={[
                    'text-sm font-medium transition-colors',
                    on ? 'text-ink' : 'text-ink-2',
                  ].join(' ')}
                >
                  {s.label}
                </span>
              </div>
              <div className="mt-0.5 pl-4 font-mono text-[0.68rem] text-ink-faint">
                {s.sub} · n={s.n}
              </div>
            </button>
          );
        })}
      </div>

      {/* chart */}
      <div className="rounded-2xl border border-line bg-card p-3 sm:p-5">
        <svg
          viewBox={`0 0 560 ${height}`}
          className="w-full"
          role="img"
          aria-label={`R-squared by soil property against ${series.label}`}
        >
          {/* gridlines */}
          {[-0.3, 0, 0.25, 0.5, 0.75, 1.0].map((g) => {
            const x = scale(g, LO, HI, X0, X1);
            const isZero = g === 0;
            return (
              <g key={g}>
                <line
                  x1={x}
                  x2={x}
                  y1={TOP - 6}
                  y2={height - TOP + 2}
                  stroke={isZero ? 'var(--color-line-strong)' : 'var(--color-line)'}
                  strokeWidth={isZero ? 1.4 : 1}
                  strokeDasharray={isZero ? '' : '2 4'}
                />
                <text
                  x={x}
                  y={height - 2}
                  textAnchor="middle"
                  className="fill-ink-faint"
                  style={{ fontFamily: 'var(--font-mono)', fontSize: 10 }}
                >
                  {g}
                </text>
              </g>
            );
          })}

          {TARGETS.map((t, i) => {
            const v = series.r2[t.key as Target];
            const y = TOP + i * ROW_H;
            const barY = y + 8;
            const barH = ROW_H - 26;
            const valX = scale(v, LO, HI, X0, X1);
            const neg = v < 0;
            const x = Math.min(zeroX, valX);
            const w = Math.abs(valX - zeroX);
            const color = neg
              ? 'var(--color-clay)'
              : series.tone === 'good'
                ? 'var(--color-moss-500)'
                : series.tone === 'warn'
                  ? 'var(--color-moss-400)'
                  : 'var(--color-clay-soft)';
            return (
              <g key={t.key}>
                <text
                  x={X0 - 12}
                  y={barY + barH / 2 + 4}
                  textAnchor="end"
                  className="fill-ink-2"
                  style={{ fontFamily: 'var(--font-sans)', fontSize: 13, fontWeight: 500 }}
                >
                  {t.label}
                </text>
                {/* track */}
                <rect
                  x={X0}
                  y={barY}
                  width={X1 - X0}
                  height={barH}
                  rx={5}
                  className="fill-paper-2"
                />
                <motion.rect
                  y={barY}
                  height={barH}
                  rx={5}
                  fill={color}
                  initial={false}
                  animate={{ x, width: w }}
                  transition={{ duration: 0.7, ease: easeOut }}
                />
                <motion.text
                  y={barY + barH / 2 + 4}
                  className="fill-ink"
                  style={{ fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 600 }}
                  textAnchor={neg ? 'end' : 'start'}
                  initial={false}
                  animate={{ x: neg ? x - 8 : valX + 8 }}
                  transition={{ duration: 0.7, ease: easeOut }}
                >
                  {fmt2(v)}
                </motion.text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* narrative */}
      <motion.p
        key={series.key}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: easeOut }}
        className="mt-4 max-w-2xl text-[0.95rem] leading-relaxed text-ink-2"
      >
        {NARRATIVE[series.key]}
      </motion.p>
      <p className="mt-3 font-mono text-[0.68rem] text-ink-faint">
        R² on held-out data · ‹0 means worse than predicting the mean ·
        sources: cv_jepa.json, verification_jepa.json, cv_jepa_kssl.json
      </p>
    </div>
  );
}
