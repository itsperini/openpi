export type InferenceBackend = "local" | "modal" | "vm";

export interface EpisodeSummary {
  episode_id: string;
  created_at: string;
  instruction: string;
  success: boolean;
  duration_seconds: number;
  metadata: string;
  inference_backend?: InferenceBackend;
  inference_transport?: string;
}

export interface EpisodeIndex {
  schema_version: number;
  updated_at: string;
  episodes: EpisodeSummary[];
}

export interface EpisodeMetadata {
  schema_version?: number;
  episode_id: string;
  created_at: string;
  serving?: {
    backend: InferenceBackend;
    transport: string;
    endpoint: string;
  };
  task: { suite: string; id: number; trial: number; instruction: string };
  result: { success: boolean; error: string | null };
  timeline: { control_hz: number; frame_count: number; duration_seconds: number };
  policy: {
    name: string;
    checkpoint: string;
    replan_steps: number;
    action_horizon: number;
    inference_count: number;
    trace_count?: number;
    mean_inference_ms: number | null;
    server_metadata: Record<string, unknown>;
  };
  pipeline?: {
    model: ModelPipelineMetadata;
    inputs: Array<{ id: string; label: string; role: PipelineRole }>;
    policy_state_labels: string[];
  };
  series: { joints: string[]; actions: string[] };
  artifacts: {
    trajectory: string;
    videos: { mujoco: string; agent: string; wrist: string };
  };
}

export type PipelineRole = "policy_input" | "telemetry_only" | "masked_padding" | "transported_input";

export interface ModelPipelineMetadata {
  config?: string;
  model_type?: string;
  action_horizon?: number;
  model_action_dim?: number;
  max_prompt_tokens?: number;
  pi05?: boolean;
  discrete_state_input?: boolean;
  vlm_variant?: string;
  action_expert_variant?: string;
  flow_steps?: number;
}

export interface DebugInputSummary {
  raw: {
    prompt: string;
    state: number[];
    state_shape: number[];
    images: Record<string, { shape: number[]; dtype: string }>;
  };
  model: {
    state: number[];
    state_shape: number[];
    prompt_tokens: number[];
    prompt_token_count: number;
    prompt_token_capacity: number;
    images: Record<string, { shape: number[]; dtype: string; range: number[]; mask: boolean }>;
  };
}

export interface FlowTrace {
  supported: boolean;
  model: ModelPipelineMetadata;
  input: DebugInputSummary;
  flow: {
    timesteps: number[];
    action_states: number[][][];
    velocities: number[][][];
  } | null;
  output_action_dim: number;
}

export interface TrajectoryRecord {
  step: number;
  timestamp: number;
  observation: {
    joint_position: number[];
    joint_velocity: number[];
    ee_position: number[];
    ee_quaternion: number[];
    gripper_position: number[];
    gripper_velocity: number[];
  };
  policy_input?: {
    state: number[];
    state_labels: string[];
    prompt: string;
  };
  policy: {
    executed_action: number[];
    chunk_id: number;
    chunk_step: number;
    action_chunk: number[][] | null;
    inference_ms: number | null;
    server_timing: Record<string, number>;
    policy_timing: Record<string, number>;
    debug_trace?: FlowTrace | null;
  };
  result: { reward: number; done: boolean };
}

export interface LoadedEpisode {
  metadata: EpisodeMetadata;
  trajectory: TrajectoryRecord[];
  baseUrl: string;
}
