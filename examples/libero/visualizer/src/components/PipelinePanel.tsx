import { Ban, Binary, Camera, Eye, MessageSquareText, Network, Radio, ScanLine } from "lucide-react";

import type { EpisodeMetadata, FlowTrace, PipelineRole, TrajectoryRecord } from "../types";

interface Props {
  metadata: EpisodeMetadata;
  current: TrajectoryRecord;
  trace: FlowTrace | null;
}

const roleDetails: Record<PipelineRole, { label: string; className: string }> = {
  policy_input: { label: "used by policy", className: "policy-input" },
  telemetry_only: { label: "telemetry only", className: "telemetry-only" },
  masked_padding: { label: "masked padding", className: "masked-input" },
  transported_input: { label: "interface input", className: "transported-input" },
};

const icons: Record<string, typeof Camera> = {
  mujoco: Eye,
  agent: Camera,
  wrist: Camera,
  right_wrist: Ban,
  joint_state: Radio,
  policy_state: Binary,
  prompt: MessageSquareText,
};

const fallbackInputs: NonNullable<EpisodeMetadata["pipeline"]>["inputs"] = [
  { id: "mujoco", label: "MuJoCo front view", role: "telemetry_only" },
  { id: "agent", label: "Agent camera", role: "policy_input" },
  { id: "wrist", label: "Wrist camera", role: "policy_input" },
  { id: "right_wrist", label: "Third camera slot", role: "masked_padding" },
  { id: "joint_state", label: "Joint qpos / qvel", role: "telemetry_only" },
  { id: "policy_state", label: "8-D end-effector state", role: "transported_input" },
  { id: "prompt", label: "Language instruction", role: "policy_input" },
];

function vector(values: number[] | undefined, limit = 8) {
  if (!values?.length) return "—";
  const visible = values.slice(0, limit).map((value) => value.toFixed(3)).join(", ");
  return `[${visible}${values.length > limit ? `, … +${values.length - limit}` : ""}]`;
}

export function PipelinePanel({ metadata, current, trace }: Props) {
  const pipeline = metadata.pipeline;
  const model = trace?.model ?? pipeline?.model ?? {};
  const input = trace?.input;
  const stateConditioned = Boolean(model.discrete_state_input);

  return (
    <section className="panel learning-panel pipeline-panel">
      <div className="panel-heading learning-heading">
        <div><h2>What actually reaches π₀.₅?</h2><p>Data provenance for this exact LIBERO checkpoint</p></div>
        <Network size={18} />
      </div>

      <div className="model-route" aria-label="Model architecture route">
        <span><ScanLine /> SigLIP patches</span><i>→</i><span>{model.vlm_variant ?? "gemma_2b"} VLM</span><i>→</i><span>{model.action_expert_variant ?? "gemma_300m"} expert</span><i>→</i><span>{model.flow_steps ?? 10} flow steps</span>
      </div>

      <ul className="provenance-list">
        {(pipeline?.inputs ?? fallbackInputs).map((item) => {
          const Icon = icons[item.id] ?? Radio;
          const detail = roleDetails[item.role];
          const special = item.id === "policy_state" && !stateConditioned;
          return (
            <li key={item.id}>
              <Icon />
              <span><strong>{item.label}</strong><small>{special ? "sent and normalized, but not consumed by this model configuration" : detail.label}</small></span>
              <em className={special ? "not-conditioned" : detail.className}>{special ? "not conditioned" : detail.label}</em>
            </li>
          );
        })}
      </ul>

      <div className="tensor-inspector">
        <div><small>Raw policy state · {current.policy_input?.state.length ?? input?.raw.state_shape[0] ?? 0}D</small><code>{vector(current.policy_input?.state ?? input?.raw.state)}</code></div>
        <div><small>Normalized + padded model state · {input?.model.state_shape[0] ?? model.model_action_dim ?? 32}D</small><code>{vector(input?.model.state)}</code></div>
        <div><small>Language tokens</small><code>{input ? `${input.model.prompt_token_count} active / ${input.model.prompt_token_capacity} capacity` : "trace unavailable"}</code></div>
        <div><small>Action path</small><code>{model.action_horizon ?? metadata.policy.action_horizon}×{model.model_action_dim ?? 32} → {metadata.policy.action_horizon}×7 → execute {metadata.policy.replan_steps}</code></div>
      </div>
    </section>
  );
}
