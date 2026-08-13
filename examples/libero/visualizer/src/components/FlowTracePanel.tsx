import { ChevronLeft, ChevronRight, Waves } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import type { FlowTrace } from "../types";

interface Props {
  trace: FlowTrace | null;
  sourceStep: number | null;
  labels: string[];
}

function norm(values: number[]) {
  return Math.sqrt(values.reduce((total, value) => total + value * value, 0));
}

function matrixNorm(values: number[][]) {
  return norm(values.flat());
}

function linePath(values: number[], width: number, height: number, max: number) {
  return values
    .map((value, index) => {
      const x = 12 + (index / Math.max(values.length - 1, 1)) * (width - 24);
      const y = height - 18 - (value / Math.max(max, 1e-6)) * (height - 34);
      return `${index ? "L" : "M"}${x},${y}`;
    })
    .join(" ");
}

export function FlowTracePanel({ trace, sourceStep, labels }: Props) {
  const states = trace?.flow?.action_states ?? [];
  const velocities = trace?.flow?.velocities ?? [];
  const timesteps = trace?.flow?.timesteps ?? [];
  const outputDims = Math.min(trace?.output_action_dim ?? labels.length, labels.length);
  const [flowStep, setFlowStep] = useState(Math.max(states.length - 1, 0));

  useEffect(() => setFlowStep(Math.max(states.length - 1, 0)), [sourceStep, states.length]);

  const metrics = useMemo(() => {
    const stateNorms = states.map(matrixNorm);
    const velocityNorms = velocities.map(matrixNorm);
    const updateNorms = states.slice(1).map((state, index) =>
      matrixNorm(state.map((row, rowIndex) => row.map((value, dim) => value - states[index][rowIndex][dim])))
    );
    return { stateNorms, velocityNorms, updateNorms };
  }, [states, velocities]);

  if (!trace?.supported || !trace.flow || states.length === 0) {
    return (
      <section className="panel learning-panel empty-learning">
        <Waves />
        <div><h2>Flow-matching trace</h2><p>This episode predates internal tracing. Record another rollout to inspect the sampler.</p></div>
      </section>
    );
  }

  const current = states[flowStep];
  const values = states.flat(2).filter((_, index) => index % (trace.model.model_action_dim ?? 32) < outputDims);
  const maxAbs = Math.max(...values.map(Math.abs), 1e-6);
  const chartMax = Math.max(...metrics.stateNorms, ...metrics.velocityNorms, ...metrics.updateNorms, 1e-6);
  const chartWidth = 500;
  const chartHeight = 132;

  return (
    <section className="panel learning-panel flow-panel">
      <div className="panel-heading learning-heading">
        <div><h2>Flow-matching microscope</h2><p>Real normalized action state as Gaussian noise becomes a 10-step plan</p></div>
        <span className="trace-source">inference at episode step {sourceStep}</span>
      </div>

      <div className="flow-controls">
        <button type="button" aria-label="Previous flow step" onClick={() => setFlowStep(Math.max(0, flowStep - 1))}><ChevronLeft /></button>
        <input aria-label="Flow integration step" type="range" min="0" max={states.length - 1} value={flowStep} onChange={(event) => setFlowStep(Number(event.target.value))} />
        <button type="button" aria-label="Next flow step" onClick={() => setFlowStep(Math.min(states.length - 1, flowStep + 1))}><ChevronRight /></button>
        <strong>stage {flowStep}/{states.length - 1}</strong>
        <span>t = {(timesteps[flowStep] ?? 0).toFixed(1)} · {flowStep === 0 ? "Gaussian noise" : flowStep === states.length - 1 ? "final action" : "Euler update"}</span>
      </div>

      <div className="flow-content">
        <div className="action-heatmap" style={{ gridTemplateColumns: `56px repeat(${outputDims}, minmax(42px, 1fr))` }} role="img" aria-label={`Action-state heatmap at flow stage ${flowStep}`}>
          <span />
          {labels.slice(0, outputDims).map((label) => <strong key={label}>{label}</strong>)}
          {current.map((row, actionIndex) => (
            <div className="heatmap-row" key={actionIndex}>
              <span>a+{actionIndex}</span>
              {row.slice(0, outputDims).map((value, dim) => {
                const strength = 12 + Math.round((Math.abs(value) / maxAbs) * 78);
                const color = value >= 0 ? "var(--cyan)" : "var(--pink)";
                return <output key={dim} aria-label={`${labels[dim]} action ${actionIndex}: ${value.toFixed(3)}`} style={{ background: `color-mix(in srgb, ${color} ${strength}%, var(--surface-raised))` }}>{value.toFixed(2)}</output>;
              })}
            </div>
          ))}
        </div>

        <div className="flow-norms">
          <div className="norm-legend"><span className="state-norm">action state ‖xₜ‖</span><span className="velocity-norm">velocity ‖vₜ‖</span><span className="update-norm">update ‖Δx‖</span></div>
          <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} aria-label="Flow state, velocity and update norms">
            {[0, 0.5, 1].map((fraction) => <line key={fraction} className="grid-line" x1="12" x2={chartWidth - 12} y1={chartHeight - 18 - fraction * (chartHeight - 34)} y2={chartHeight - 18 - fraction * (chartHeight - 34)} />)}
            <path className="state-norm-line" d={linePath(metrics.stateNorms, chartWidth, chartHeight, chartMax)} />
            <path className="velocity-norm-line" d={linePath([...metrics.velocityNorms, metrics.velocityNorms.at(-1) ?? 0], chartWidth, chartHeight, chartMax)} />
            <path className="update-norm-line" d={linePath([0, ...metrics.updateNorms], chartWidth, chartHeight, chartMax)} />
            <line className="flow-cursor" x1={12 + (flowStep / (states.length - 1)) * (chartWidth - 24)} x2={12 + (flowStep / (states.length - 1)) * (chartWidth - 24)} y1="8" y2={chartHeight - 18} />
            <text className="axis-label" x="12" y={chartHeight - 3}>noise · t=1</text>
            <text className="axis-label" x={chartWidth - 12} y={chartHeight - 3} textAnchor="end">action · t=0</text>
          </svg>
          <dl className="flow-values">
            <div><dt>model tensor</dt><dd>{current.length} × {current[0]?.length ?? 0}</dd></div>
            <div><dt>visible output</dt><dd>{current.length} × {outputDims}</dd></div>
            <div><dt>action-state norm</dt><dd>{metrics.stateNorms[flowStep]?.toFixed(3)}</dd></div>
            <div><dt>velocity norm</dt><dd>{metrics.velocityNorms[Math.min(flowStep, metrics.velocityNorms.length - 1)]?.toFixed(3) ?? "—"}</dd></div>
          </dl>
        </div>
      </div>
    </section>
  );
}
