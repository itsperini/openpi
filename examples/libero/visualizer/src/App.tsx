import { Activity, Check, ChevronLeft, ChevronRight, CircleStop, Gauge, Pause, Play, RotateCcw, Video } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { EndEffectorPlot } from "./components/EndEffectorPlot";
import { TelemetryChart } from "./components/TelemetryChart";
import type { EpisodeIndex, EpisodeMetadata, LoadedEpisode, TrajectoryRecord } from "./types";


function asset(path: string) {
  return `/${path.replace(/^\//, "")}`;
}

async function loadEpisode(summaryPath: string): Promise<LoadedEpisode> {
  const metadata = await fetch(asset(summaryPath)).then((response) => response.json()) as EpisodeMetadata;
  const baseUrl = summaryPath.slice(0, summaryPath.lastIndexOf("/"));
  const text = await fetch(asset(`${baseUrl}/${metadata.artifacts.trajectory}`)).then((response) => response.text());
  const trajectory = text.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line) as TrajectoryRecord);
  return { metadata, trajectory, baseUrl };
}

function formatVector(values: number[] | undefined, digits = 3) {
  if (!values) return "—";
  return `[${values.map((value) => value.toFixed(digits)).join(", ")}]`;
}

function CameraPanel({ title, label, src, videoRef }: { title: string; label: string; src: string; videoRef: (node: HTMLVideoElement | null) => void }) {
  return (
    <section className="panel camera-panel">
      <div className="camera-label"><span>{title}</span><small>{label}</small></div>
      <video ref={videoRef} src={src} muted playsInline preload="auto" />
    </section>
  );
}

export default function App() {
  const [index, setIndex] = useState<EpisodeIndex | null>(null);
  const [selected, setSelected] = useState("");
  const [episode, setEpisode] = useState<LoadedEpisode | null>(null);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState("");
  const videos = useRef<Record<string, HTMLVideoElement | null>>({});
  const animation = useRef(0);

  useEffect(() => {
    fetch(asset("index.json"))
      .then((response) => {
        if (!response.ok) throw new Error("No recorded episodes found. Run the LIBERO policy once first.");
        return response.json() as Promise<EpisodeIndex>;
      })
      .then((data) => {
        setIndex(data);
        if (data.episodes[0]) setSelected(data.episodes[0].metadata);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setPlaying(false);
    setStep(0);
    loadEpisode(selected).then(setEpisode).catch((reason: Error) => setError(reason.message));
  }, [selected]);

  const syncStep = useCallback(() => {
    const main = videos.current.mujoco;
    if (main && episode) {
      const nextStep = Math.min(episode.trajectory.length - 1, Math.round(main.currentTime * episode.metadata.timeline.control_hz));
      setStep(nextStep);
      Object.entries(videos.current).forEach(([name, video]) => {
        if (name !== "mujoco" && video && Math.abs(video.currentTime - main.currentTime) > 0.08) video.currentTime = main.currentTime;
      });
      if (!main.paused && !main.ended) animation.current = requestAnimationFrame(syncStep);
      else setPlaying(false);
    }
  }, [episode]);

  useEffect(() => () => cancelAnimationFrame(animation.current), []);

  const seek = useCallback((nextStep: number) => {
    if (!episode) return;
    const clamped = Math.max(0, Math.min(episode.trajectory.length - 1, nextStep));
    const time = clamped / episode.metadata.timeline.control_hz;
    Object.values(videos.current).forEach((video) => { if (video) video.currentTime = time; });
    setStep(clamped);
  }, [episode]);

  async function togglePlayback() {
    if (!episode) return;
    if (playing) {
      Object.values(videos.current).forEach((video) => video?.pause());
      setPlaying(false);
      return;
    }
    if (step >= episode.trajectory.length - 1) seek(0);
    await Promise.all(Object.values(videos.current).filter(Boolean).map((video) => video!.play()));
    setPlaying(true);
    cancelAnimationFrame(animation.current);
    animation.current = requestAnimationFrame(syncStep);
  }

  const current = episode?.trajectory[step];
  const latestPlan = useMemo(() => {
    if (!episode) return null;
    for (let index = step; index >= 0; index -= 1) {
      if (episode.trajectory[index].policy.action_chunk) {
        return { start: index, values: episode.trajectory[index].policy.action_chunk! };
      }
    }
    return null;
  }, [episode, step]);

  if (error) return <main className="empty"><Activity /><h1>Episode data unavailable</h1><p>{error}</p></main>;
  if (!episode || !index || !current) return <main className="empty"><Activity className="spin" /><h1>Loading episode telemetry</h1></main>;

  const metadata = episode.metadata;
  const videoUrl = (name: keyof EpisodeMetadata["artifacts"]["videos"]) => asset(`${episode.baseUrl}/${metadata.artifacts.videos[name]}`);
  const jointPosition = episode.trajectory.map((record) => record.observation.joint_position);
  const jointVelocity = episode.trajectory.map((record) => record.observation.joint_velocity);
  const eePosition = episode.trajectory.map((record) => record.observation.ee_position);
  const actions = episode.trajectory.map((record) => record.policy.executed_action);
  const inference = episode.trajectory.slice(0, step + 1).reverse().find((record) => record.policy.inference_ms !== null)?.policy.inference_ms;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark"><Activity size={18} /></span><div><strong>OpenPI Inspector</strong><small>LIBERO · Franka Panda</small></div></div>
        <div className="episode-picker">
          <label htmlFor="episode">Episode</label>
          <select id="episode" value={selected} onChange={(event) => setSelected(event.target.value)}>
            {index.episodes.map((item) => <option key={item.episode_id} value={item.metadata}>{item.episode_id}</option>)}
          </select>
        </div>
        <div className={`result ${metadata.result.success ? "success" : "failure"}`}>
          {metadata.result.success ? <Check size={15} /> : <CircleStop size={15} />}{metadata.result.success ? "Task success" : "Incomplete"}
        </div>
      </header>

      <section className="mission-bar">
        <div><small>Instruction</small><h1>{metadata.task.instruction}</h1></div>
        <div className="mission-stats"><span><small>Step</small>{step} / {metadata.timeline.frame_count - 1}</span><span><small>Time</small>{current.timestamp.toFixed(2)} s</span><span><small>Chunk</small>{current.policy.chunk_id}:{current.policy.chunk_step}</span></div>
      </section>

      <section className="media-grid">
        <section className="panel main-video">
          <div className="camera-label"><span><Video size={15} /> MuJoCo overview</span><small>frontview</small></div>
          <video ref={(node) => { videos.current.mujoco = node; }} src={videoUrl("mujoco")} muted playsInline preload="auto" onEnded={() => setPlaying(false)} />
        </section>
        <div className="camera-stack">
          <CameraPanel title="Policy camera" label="agentview · 224 × 224" src={videoUrl("agent")} videoRef={(node) => { videos.current.agent = node; }} />
          <CameraPanel title="Wrist camera" label="eye-in-hand · 224 × 224" src={videoUrl("wrist")} videoRef={(node) => { videos.current.wrist = node; }} />
        </div>
      </section>

      <section className="timeline panel">
        <button type="button" aria-label="Previous frame" onClick={() => seek(step - 1)}><ChevronLeft /></button>
        <button type="button" className="play" aria-label={playing ? "Pause" : "Play"} onClick={togglePlayback}>{playing ? <Pause /> : <Play />}</button>
        <button type="button" aria-label="Next frame" onClick={() => seek(step + 1)}><ChevronRight /></button>
        <input aria-label="Episode timeline" type="range" min="0" max={metadata.timeline.frame_count - 1} value={step} onChange={(event) => seek(Number(event.target.value))} />
        <button type="button" aria-label="Restart episode" onClick={() => seek(0)}><RotateCcw /></button>
        <time>{current.timestamp.toFixed(2)} / {metadata.timeline.duration_seconds.toFixed(2)} s</time>
      </section>

      <section className="charts-grid">
        <TelemetryChart title="Joint position" subtitle="Panda arm configuration · radians" values={jointPosition} labels={metadata.series.joints} activeStep={step} onSeek={seek} />
        <TelemetryChart title="Joint velocity" subtitle="Angular velocity · radians / second" values={jointVelocity} labels={metadata.series.joints} activeStep={step} onSeek={seek} />
      </section>

      <section className="analysis-grid">
        <EndEffectorPlot values={eePosition} activeStep={step} onSeek={seek} />
        <TelemetryChart title="Policy actions" subtitle="Solid: executed · dashed: latest predicted chunk" values={actions} labels={metadata.series.actions} activeStep={step} onSeek={seek} projection={latestPlan} />
      </section>

      <section className="inspectors">
        <section className="panel inspector">
          <div className="panel-heading"><div><h2>Observation</h2><p>Exact state sent at this timestep</p></div><Activity size={18} /></div>
          <dl>
            <div><dt>joint position</dt><dd>{formatVector(current.observation.joint_position)}</dd></div>
            <div><dt>joint velocity</dt><dd>{formatVector(current.observation.joint_velocity)}</dd></div>
            <div><dt>end effector xyz</dt><dd>{formatVector(current.observation.ee_position)}</dd></div>
            <div><dt>end effector quat</dt><dd>{formatVector(current.observation.ee_quaternion)}</dd></div>
            <div><dt>gripper</dt><dd>{formatVector(current.observation.gripper_position)}</dd></div>
          </dl>
        </section>
        <section className="panel inspector">
          <div className="panel-heading"><div><h2>Policy</h2><p>Plan, execution and timing</p></div><Gauge size={18} /></div>
          <dl>
            <div><dt>checkpoint</dt><dd>{metadata.policy.checkpoint}</dd></div>
            <div><dt>last inference</dt><dd>{inference?.toFixed(1) ?? "—"} ms</dd></div>
            <div><dt>mean inference</dt><dd>{metadata.policy.mean_inference_ms?.toFixed(1) ?? "—"} ms</dd></div>
            <div><dt>action horizon</dt><dd>{metadata.policy.action_horizon} · execute {metadata.policy.replan_steps}</dd></div>
            <div><dt>executed action</dt><dd>{formatVector(current.policy.executed_action)}</dd></div>
          </dl>
        </section>
      </section>
    </main>
  );
}
