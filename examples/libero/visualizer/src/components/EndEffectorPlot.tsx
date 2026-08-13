interface Props {
  values: number[][];
  activeStep: number;
  onSeek: (step: number) => void;
}

export function EndEffectorPlot({ values, activeStep, onSeek }: Props) {
  const width = 430;
  const height = 250;
  const padding = 30;
  const xs = values.map((value) => value[0]);
  const ys = values.map((value) => value[1]);
  const extent = (items: number[]) => {
    const min = Math.min(...items);
    const max = Math.max(...items);
    const pad = Math.max((max - min) * 0.15, 0.01);
    return [min - pad, max + pad];
  };
  const xDomain = extent(xs);
  const yDomain = extent(ys);
  const x = (value: number) => padding + ((value - xDomain[0]) / (xDomain[1] - xDomain[0])) * (width - padding * 2);
  const y = (value: number) => height - padding - ((value - yDomain[0]) / (yDomain[1] - yDomain[0])) * (height - padding * 2);
  const path = values.map((value, index) => `${index ? "L" : "M"}${x(value[0])},${y(value[1])}`).join(" ");
  const current = values[activeStep] ?? values[0] ?? [0, 0, 0];

  function seek(event: React.PointerEvent<SVGSVGElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - bounds.left) / bounds.width) * width;
    const py = ((event.clientY - bounds.top) / bounds.height) * height;
    let nearest = 0;
    let distance = Number.POSITIVE_INFINITY;
    values.forEach((value, index) => {
      const candidate = (x(value[0]) - px) ** 2 + (y(value[1]) - py) ** 2;
      if (candidate < distance) {
        distance = candidate;
        nearest = index;
      }
    });
    onSeek(nearest);
  }

  return (
    <section className="panel ee-panel">
      <div className="panel-heading"><div><h2>End-effector trajectory</h2><p>Top-down XY path · click to seek</p></div><strong>z {current[2].toFixed(3)} m</strong></div>
      <svg className="ee-chart" viewBox={`0 0 ${width} ${height}`} onPointerDown={seek} aria-label="End-effector XY trajectory">
        <rect className="plot-frame" x={padding} y={padding} width={width - padding * 2} height={height - padding * 2} />
        <path d={path} fill="none" stroke="var(--cyan)" strokeWidth="2.5" opacity="0.72" />
        {values.map((value, index) => index % 8 === 0 && <circle key={index} cx={x(value[0])} cy={y(value[1])} r="2" fill="var(--cyan)" opacity="0.45" />)}
        <circle cx={x(current[0])} cy={y(current[1])} r="7" fill="var(--cyan)" stroke="var(--surface)" strokeWidth="3" />
        <text className="axis-label" x={width / 2} y={height - 6} textAnchor="middle">x position (m)</text>
        <text className="axis-label" x="12" y={height / 2} textAnchor="middle" transform={`rotate(-90 12 ${height / 2})`}>y position (m)</text>
      </svg>
    </section>
  );
}
