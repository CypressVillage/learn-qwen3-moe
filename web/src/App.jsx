import { useEffect, useRef, useState } from "react";

import content from "./generated/content.json";
import {
  addedLineIndexes,
  checkpointIndexAfterKey,
  checkpointIndexAtReadingLine,
} from "./checkpoint-progress.js";


const checkpoints = content.checkpoints;

function Pill({ children, tone = "neutral" }) {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}

function FlowMap() {
  const stages = [
    ["CONFIG", "ready"],
    ["WEIGHTS", "ready"],
    ["TOKENIZER", "waiting"],
    ["MODEL", "waiting"],
    ["LOGITS", "waiting"],
    ["GENERATE", "waiting"],
  ];
  return (
    <div className="flow-map" aria-label="完整推理数据流">
      {stages.map(([label, status], index) => (
        <div className="flow-stage" key={label}>
          <span className={`status-dot ${status}`} />
          <span>{String(index + 1).padStart(2, "0")}</span>
          <strong>{label}</strong>
          <small>{status === "ready" ? "ONLINE" : "LOCKED"}</small>
        </div>
      ))}
    </div>
  );
}

function useReadingCheckpoint() {
  const [checkpointIndex, setCheckpointIndex] = useState(0);

  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      const readingLine = window.innerHeight * 0.42;
      const anchors = checkpoints.map((item) =>
        document.querySelector(`[data-checkpoint="${item.id}"]`),
      );
      const nextIndex = checkpointIndexAtReadingLine(
        anchors.map((anchor) => anchor?.getBoundingClientRect().top ?? Infinity),
        readingLine,
      );
      setCheckpointIndex(nextIndex);
      anchors.forEach((anchor, index) => anchor?.classList.toggle("active", index === nextIndex));
    };
    const scheduleUpdate = () => {
      if (!frame) frame = window.requestAnimationFrame(update);
    };

    scheduleUpdate();
    window.addEventListener("scroll", scheduleUpdate, { passive: true });
    window.addEventListener("resize", scheduleUpdate);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", scheduleUpdate);
      window.removeEventListener("resize", scheduleUpdate);
    };
  }, []);

  return checkpointIndex;
}

function RepositoryView({ checkpoint, previousCheckpoint }) {
  const [filePath, setFilePath] = useState(checkpoint.active_file);
  const codePanelRef = useRef(null);

  useEffect(() => {
    setFilePath(checkpoint.active_file);
  }, [checkpoint.active_file, checkpoint.id]);

  useEffect(() => {
    const panel = codePanelRef.current;
    const focused = panel?.querySelector(".code-line.focused");
    if (panel && focused) {
      panel.scrollTo({
        top: Math.max(0, focused.offsetTop - panel.clientHeight * 0.28),
        behavior: "smooth",
      });
    } else if (panel) {
      panel.scrollTop = 0;
    }
  }, [checkpoint.id, filePath]);

  const file = checkpoint.repository_snapshot.files.find((item) => item.path === filePath);
  const previousFile = previousCheckpoint?.repository_snapshot.files.find(
    (item) => item.path === filePath,
  );
  const lines = file.content
    ? (file.content.endsWith("\n") ? file.content.slice(0, -1) : file.content).split("\n")
    : [];
  const addedLines = new Set(addedLineIndexes(previousFile?.content ?? "", file.content));

  return (
    <div className="repo-view">
      <div className="file-list" aria-label="累计 checkpoint 文件">
        {checkpoint.repository_snapshot.files.map((item) => (
          <button
            className={item.path === filePath ? "active" : ""}
            key={item.path}
            onClick={() => setFilePath(item.path)}
          >
            <span className={item.content ? "file-dot ready" : "file-dot"} />
            {item.path}
          </button>
        ))}
      </div>
      <div className="code-header">
        <span>{file.path}</span>
        <div>
          {file.path === checkpoint.active_file && <Pill tone="token">FOCUS</Pill>}
          <span className="line-count">{lines.length} LINES</span>
        </div>
      </div>
      <pre className={`code-panel ${lines.length === 0 ? "empty" : ""}`} ref={codePanelRef} tabIndex="0" aria-label={`${file.path} 源码`} aria-live="polite">
        {lines.length === 0 ? (
          <span className="empty-file">
            <i>EMPTY FILE</i>
            <strong>继续阅读，代码会在这里逐步出现。</strong>
            <small>Checkpoint {String(checkpoints.indexOf(checkpoint) + 1).padStart(2, "0")} / {String(checkpoints.length).padStart(2, "0")}</small>
          </span>
        ) : (
          <code key={`${checkpoint.id}-${file.path}`}>
            {lines.map((line, index) => {
              const number = index + 1;
              const focused =
                file.path === checkpoint.active_file
                && checkpoint.focus_range.start > 0
                && number >= checkpoint.focus_range.start
                && number <= checkpoint.focus_range.end;
              const added = addedLines.has(index);
              return (
                <span
                  className={`code-line ${focused ? "focused" : ""} ${added ? "added" : ""}`}
                  key={`${number}-${line}`}
                  style={added ? { animationDelay: `${Math.min(index, 30) * 12}ms` } : undefined}
                >
                  <i>{String(number).padStart(3, "0")}</i>{line || " "}
                </span>
              );
            })}
          </code>
        )}
      </pre>
    </div>
  );
}

function Lab({ checkpoint, previousCheckpoint, checkpointIndex, pinned, togglePinned, moveCheckpoint, mobileClose }) {
  return (
    <aside className="lab" aria-label="Step 01 实验台">
      <header className="lab-header">
        <div className="checkpoint-heading">
          <span>SCROLL-LINKED CHECKPOINT</span>
          <strong>{checkpoint.id}</strong>
          <small>{checkpoint.label}</small>
        </div>
        <div className="lab-actions">
          <div className="checkpoint-stepper" aria-label="Checkpoint 导航">
            <button onClick={() => moveCheckpoint(-1)} disabled={checkpointIndex === 0} aria-label="上一个 checkpoint" aria-keyshortcuts="ArrowLeft">←</button>
            <span>{String(checkpointIndex + 1).padStart(2, "0")} / {String(checkpoints.length).padStart(2, "0")}</span>
            <button onClick={() => moveCheckpoint(1)} disabled={checkpointIndex === checkpoints.length - 1} aria-label="下一个 checkpoint" aria-keyshortcuts="ArrowRight">→</button>
          </div>
          <button className={pinned ? "locked" : ""} onClick={togglePinned} aria-pressed={pinned} aria-keyshortcuts="Escape">{pinned ? "解除锁定" : "锁定"}</button>
          {mobileClose && <button onClick={mobileClose}>关闭</button>}
        </div>
      </header>
      <div className="lab-body">
        <RepositoryView checkpoint={checkpoint} previousCheckpoint={previousCheckpoint} />
      </div>
      <footer className="lab-footer"><span>SOURCE CHECKPOINT</span><span>←/→ STEP · ESC UNLOCK</span><span>INSERT-ONLY SNAPSHOTS</span></footer>
    </aside>
  );
}

export function App() {
  const readingCheckpointIndex = useReadingCheckpoint();
  const [pinnedCheckpointIndex, setPinnedCheckpointIndex] = useState(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const checkpointIndex = pinnedCheckpointIndex ?? readingCheckpointIndex;
  const checkpoint = checkpoints[checkpointIndex];
  const previousCheckpoint = checkpointIndex > 0 ? checkpoints[checkpointIndex - 1] : null;

  useEffect(() => {
    const handleKeyDown = (event) => {
      const target = event.target;
      if (
        event.ctrlKey
        || event.metaKey
        || event.altKey
        || event.shiftKey
        || target instanceof HTMLInputElement
        || target instanceof HTMLTextAreaElement
        || target instanceof HTMLSelectElement
        || target?.isContentEditable
      ) return;

      if (event.key === "Escape" && pinnedCheckpointIndex !== null) {
        event.preventDefault();
        setPinnedCheckpointIndex(null);
        return;
      }

      const nextIndex = checkpointIndexAfterKey(
        event.key,
        checkpointIndex,
        checkpoints.length,
      );
      if (nextIndex === null) return;
      event.preventDefault();
      setPinnedCheckpointIndex(nextIndex);
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [checkpointIndex, pinnedCheckpointIndex]);

  function openMobile() {
    setMobileOpen(true);
  }

  function togglePinned() {
    setPinnedCheckpointIndex((current) => current === null ? checkpointIndex : null);
  }

  function moveCheckpoint(direction) {
    setPinnedCheckpointIndex(Math.max(0, Math.min(checkpoints.length - 1, checkpointIndex + direction)));
  }

  const labProps = {
    checkpoint,
    previousCheckpoint,
    checkpointIndex,
    pinned: pinnedCheckpointIndex !== null,
    togglePinned,
    moveCheckpoint,
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/step01/"><span>Q3</span><strong>MOE INFERENCE LAB</strong></a>
        <div className="course-progress"><span>STEP 01 / 14</span><div><i /></div><b>CONFIG + WEIGHTS</b></div>
        <a className="source-button" href="https://github.com/CypressVillage/learn-qwen3-moe" aria-label="查看仓库源码">SOURCE ↗</a>
      </header>
      <main className="workspace">
        <article className="lesson-pane">
          <div className="lesson-kicker"><Pill tone="token">FOUNDATION</Pill><span>约 45 分钟 · CPU ONLY</span></div>
          <FlowMap />
          <div className="lesson-content" dangerouslySetInnerHTML={{ __html: content.lesson.html }} />
        </article>
        <Lab {...labProps} />
      </main>
      <nav className="mobile-dock" aria-label="移动端实验台入口">
        <button onClick={openMobile}>查看源码</button>
      </nav>
      {mobileOpen && <div className="mobile-scrim" onClick={() => setMobileOpen(false)} />}
      <div className={`mobile-lab ${mobileOpen ? "open" : ""}`}>
        <Lab {...labProps} mobileClose={() => setMobileOpen(false)} />
      </div>
    </div>
  );
}
