import { Layers3 } from "lucide-react";
import { useMemo } from "react";

import type { EpisodeMetadata, TrajectoryRecord } from "../types";

interface Props {
  metadata: EpisodeMetadata;
  trajectory: TrajectoryRecord[];
  activeStep: number;
  onSeek: (step: number) => void;
}

function actionNorm(action: number[]) {
  return Math.sqrt(action.reduce((total, value) => total + value * value, 0));
}

export function RecedingHorizonPanel({ metadata, trajectory, activeStep, onSeek }: Props) {
  const plans = useMemo(() => trajectory.filter((record) => record.policy.action_chunk), [trajectory]);
  const width = 1120;
  const left = 70;
  const right = 18;
  const top = 30;
  const rowHeight = 19;
  const bottom = 28;
  const height = top + plans.length * rowHeight + bottom;
  const plotWidth = width - left - right;
  const totalSteps = Math.max(trajectory.length - 1, 1);
  const x = (step: number) => left + (step / totalSteps) * plotWidth;
  const maxNorm = Math.max(...plans.flatMap((plan) => plan.policy.action_chunk!.map(actionNorm)), 1e-6);
  const tickEvery = totalSteps > 80 ? 20 : 10;
  const ticks = Array.from({ length: Math.floor(totalSteps / tickEvery) + 1 }, (_, index) => index * tickEvery);

  return (
    <section className="panel learning-panel horizon-panel">
      <div className="panel-heading learning-heading">
        <div><h2>Receding-horizon plan history</h2><p>Every row is one 10-action prediction; bright cells were executed and dim cells were discarded</p></div>
        <div className="horizon-legend"><span className="executed-key">executed</span><span className="discarded-key">discarded</span><Layers3 size={17} /></div>
      </div>
      <div className="horizon-scroll">
        <svg className="horizon-chart" viewBox={`0 0 ${width} ${height}`} style={{ minWidth: `${Math.max(720, trajectory.length * 8)}px` }} aria-label="Predicted and executed action chunks over episode time">
          {ticks.map((tick) => <g key={tick}><line className="grid-line" x1={x(tick)} x2={x(tick)} y1={top - 15} y2={height - bottom} /><text className="axis-label" x={x(tick)} y="12" textAnchor={tick === 0 ? "start" : "middle"}>{tick}</text></g>)}
          {plans.map((plan, row) => {
            const chunk = plan.policy.action_chunk!;
            const isCurrent = plan.policy.chunk_id === trajectory[activeStep]?.policy.chunk_id;
            return (
              <g key={plan.policy.chunk_id} className={isCurrent ? "current-plan" : ""}>
                <text className="plan-label" x={left - 10} y={top + row * rowHeight + 12} textAnchor="end">chunk {plan.policy.chunk_id}</text>
                {chunk.map((action, actionIndex) => {
                  const start = plan.step + actionIndex;
                  const cellWidth = Math.max(x(start + 1) - x(start) - 1, 2);
                  const executed = actionIndex < metadata.policy.replan_steps;
                  return (
                    <rect key={actionIndex} className={executed ? "plan-executed" : "plan-discarded"} x={x(start)} y={top + row * rowHeight} width={cellWidth} height={rowHeight - 4} rx="1" opacity={0.28 + (actionNorm(action) / maxNorm) * 0.72} onClick={() => onSeek(Math.min(start, totalSteps))}>
                      <title>{`chunk ${plan.policy.chunk_id}, action +${actionIndex}, ${executed ? "executed" : "discarded"}, norm ${actionNorm(action).toFixed(3)}`}</title>
                    </rect>
                  );
                })}
              </g>
            );
          })}
          <line className="horizon-cursor" x1={x(activeStep)} x2={x(activeStep)} y1={top - 18} y2={height - bottom + 2} />
          <text className="axis-label" x={width / 2} y={height - 5} textAnchor="middle">episode control step</text>
        </svg>
      </div>
    </section>
  );
}
