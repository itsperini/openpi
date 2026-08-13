import { useMemo } from "react";


const SERIES_COLORS = ["var(--cyan)", "var(--violet)", "var(--amber)", "var(--pink)", "var(--green)", "var(--blue)", "var(--red)"];

interface Props {
  title: string;
  subtitle: string;
  values: number[][];
  labels: string[];
  activeStep: number;
  onSeek: (step: number) => void;
  projection?: { start: number; values: number[][] } | null;
}

export function TelemetryChart({ title, subtitle, values, labels, activeStep, onSeek, projection }: Props) {
  const width = 820;
  const height = 210;
  const plot = { left: 48, right: 16, top: 20, bottom: 28 };
  const innerWidth = width - plot.left - plot.right;
  const innerHeight = height - plot.top - plot.bottom;

  const domain = useMemo(() => {
    const all = values.flat();
    if (projection) all.push(...projection.values.flat());
    let min = Math.min(...all);
    let max = Math.max(...all);
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [-1, 1];
    if (min === max) {
      min -= 1;
      max += 1;
    }
    const padding = (max - min) * 0.12;
    return [min - padding, max + padding];
  }, [projection, values]);

  const x = (step: number) => plot.left + (step / Math.max(values.length - 1, 1)) * innerWidth;
  const y = (value: number) => plot.top + ((domain[1] - value) / (domain[1] - domain[0])) * innerHeight;
  const line = (series: number[]) => series.map((value, index) => `${index ? "L" : "M"}${x(index)},${y(value)}`).join(" ");
  const projectedLine = (series: number[]) =>
    series.map((value, index) => `${index ? "L" : "M"}${x((projection?.start ?? 0) + index)},${y(value)}`).join(" ");

  const series = labels.map((_, seriesIndex) => values.map((row) => row[seriesIndex] ?? 0));
  const projectedSeries = labels.map((_, seriesIndex) => projection?.values.map((row) => row[seriesIndex] ?? 0) ?? []);

  function seek(event: React.PointerEvent<SVGRectElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    const fraction = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width));
    onSeek(Math.round(fraction * (values.length - 1)));
  }

  return (
    <section className="panel chart-panel">
      <div className="panel-heading">
        <div><h2>{title}</h2><p>{subtitle}</p></div>
        <div className="legend">
          {labels.map((label, index) => <span key={label}><i style={{ background: SERIES_COLORS[index % SERIES_COLORS.length] }} />{label}</span>)}
        </div>
      </div>
      <svg className="chart" viewBox={`0 0 ${width} ${height}`} aria-label={`${title} over episode time`}>
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
          <g key={fraction}>
            <line className="grid-line" x1={plot.left} x2={width - plot.right} y1={plot.top + fraction * innerHeight} y2={plot.top + fraction * innerHeight} />
            <text className="axis-label" x={plot.left - 8} y={plot.top + fraction * innerHeight + 4} textAnchor="end">
              {(domain[1] - fraction * (domain[1] - domain[0])).toFixed(2)}
            </text>
          </g>
        ))}
        {series.map((item, index) => <path key={labels[index]} d={line(item)} fill="none" stroke={SERIES_COLORS[index % SERIES_COLORS.length]} strokeWidth="1.6" />)}
        {projection && projectedSeries.map((item, index) => (
          <path key={`projection-${labels[index]}`} d={projectedLine(item)} fill="none" stroke={SERIES_COLORS[index % SERIES_COLORS.length]} strokeWidth="2" strokeDasharray="5 5" opacity="0.5" />
        ))}
        <line className="cursor-line" x1={x(activeStep)} x2={x(activeStep)} y1={plot.top} y2={height - plot.bottom} />
        <circle className="cursor-dot" cx={x(activeStep)} cy={y(series[0]?.[activeStep] ?? 0)} r="4" />
        <text className="axis-label" x={plot.left} y={height - 7}>0</text>
        <text className="axis-label" x={width - plot.right} y={height - 7} textAnchor="end">step {values.length - 1}</text>
        <rect className="chart-hit" x={plot.left} y={plot.top} width={innerWidth} height={innerHeight} onPointerDown={seek} />
      </svg>
    </section>
  );
}
