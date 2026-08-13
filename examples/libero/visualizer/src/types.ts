export interface EpisodeSummary {
  episode_id: string;
  created_at: string;
  instruction: string;
  success: boolean;
  duration_seconds: number;
  metadata: string;
}

export interface EpisodeIndex {
  schema_version: number;
  updated_at: string;
  episodes: EpisodeSummary[];
}

export interface EpisodeMetadata {
  episode_id: string;
  created_at: string;
  task: { suite: string; id: number; trial: number; instruction: string };
  result: { success: boolean; error: string | null };
  timeline: { control_hz: number; frame_count: number; duration_seconds: number };
  policy: {
    name: string;
    checkpoint: string;
    replan_steps: number;
    action_horizon: number;
    inference_count: number;
    mean_inference_ms: number | null;
    server_metadata: Record<string, unknown>;
  };
  series: { joints: string[]; actions: string[] };
  artifacts: {
    trajectory: string;
    videos: { mujoco: string; agent: string; wrist: string };
  };
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
  policy: {
    executed_action: number[];
    chunk_id: number;
    chunk_step: number;
    action_chunk: number[][] | null;
    inference_ms: number | null;
    server_timing: Record<string, number>;
    policy_timing: Record<string, number>;
  };
  result: { reward: number; done: boolean };
}

export interface LoadedEpisode {
  metadata: EpisodeMetadata;
  trajectory: TrajectoryRecord[];
  baseUrl: string;
}
