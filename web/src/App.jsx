import { useEffect, useRef, useState } from "react";
import Prism from "prismjs";
import "prismjs/components/prism-python";

import content from "./generated/content.json";
import {
  addedLineIndexes,
  checkpointIndexAfterKey,
  checkpointIndexAtReadingLine,
} from "./checkpoint-progress.js";


const baseUrl = import.meta.env.BASE_URL;
const stepPresentation = {
  step00: {
    number: "00",
    progress: 0,
    label: "QWEN3 MOE OVERVIEW",
    title: "Qwen3 MoE 概览",
    summary: "完整推理地图与源码骨架",
    labLabel: "Step 00 架构地图实验台",
    kicker: "ORIENTATION",
    duration: "约 30 分钟 · CPU ONLY",
  },
  step01: {
    number: "01",
    progress: 7.14,
    label: "CONFIG + WEIGHTS",
    title: "配置与权重目录",
    summary: "读取模型配置与 Safetensors 索引",
    labLabel: "Step 01 实验台",
    kicker: "FOUNDATION",
    duration: "约 45 分钟 · CPU ONLY",
  },
  step02: {
    number: "02",
    progress: 14.28,
    label: "BYTE-LEVEL BPE",
    title: "Tokenizer",
    summary: "把文本编码成 Qwen3 token IDs",
    labLabel: "Step 02 Tokenizer 实验台",
    kicker: "TEXT INPUT",
    duration: "约 50 分钟 · CPU ONLY",
  },
  step03: {
    number: "03",
    progress: 21.42,
    label: "BASIC NUMPY LAYERS",
    title: "基础层",
    summary: "Embedding、RMSNorm 与 Linear",
    labLabel: "Step 03 基础层实验台",
    kicker: "TENSOR ENTRY",
    duration: "约 45 分钟 · CPU ONLY",
  },
  step04: {
    number: "04",
    progress: 28.57,
    label: "ROTARY POSITION EMBEDDING",
    title: "RoPE",
    summary: "把位置信息写入 Query 和 Key",
    labLabel: "Step 04 RoPE 实验台",
    kicker: "POSITION SIGNAL",
    duration: "约 45 分钟 · CPU ONLY",
  },
  step05: {
    number: "05",
    progress: 35.71,
    label: "GROUPED-QUERY ATTENTION",
    title: "GQA Attention",
    summary: "让每个 token 只读取已经出现的上下文",
    labLabel: "Step 05 GQA Attention 实验台",
    kicker: "CONTEXT READING",
    duration: "约 60 分钟 · CPU ONLY",
  },
  step06: {
    number: "06",
    progress: 42.85,
    label: "SPARSE MIXTURE OF EXPERTS",
    title: "Sparse MoE",
    summary: "为每个 token 选择并合并 top-k experts",
    labLabel: "Step 06 Sparse MoE 实验台",
    kicker: "EXPERT ROUTING",
    duration: "约 60 分钟 · CPU ONLY",
  },
  step07: {
    number: "07",
    progress: 50,
    label: "DECODER LAYER ASSEMBLY",
    title: "Decoder Layer",
    summary: "组装 Attention、MoE 与两条 residual",
    labLabel: "Step 07 Decoder Layer 实验台",
    kicker: "LAYER ASSEMBLY",
    duration: "约 50 分钟 · CPU ONLY",
  },
};

const savedTheme = window.localStorage.getItem("qwen3-moe-theme");
const initialTheme = savedTheme === "light" || savedTheme === "dark"
  ? savedTheme
  : window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
document.documentElement.dataset.theme = initialTheme;
document.documentElement.style.colorScheme = initialTheme;

function Pill({ children, tone = "neutral" }) {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}

function useReadingCheckpoint(checkpoints) {
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
  }, [checkpoints]);

  return checkpointIndex;
}

function buildFileTree(files) {
  const root = { directories: new Map(), files: [] };

  files.forEach((file) => {
    const parts = file.path.split("/");
    const fileName = parts.pop();
    let directory = root;

    parts.forEach((part) => {
      if (!directory.directories.has(part)) {
        directory.directories.set(part, { directories: new Map(), files: [] });
      }
      directory = directory.directories.get(part);
    });
    directory.files.push({ ...file, name: fileName });
  });

  return root;
}

function FileTreeLevel({ node, activePath, previousPaths, onSelect }) {
  return (
    <ul>
      {[...node.directories.entries()].map(([name, child]) => (
        <li className="tree-directory" key={name}>
          <details open>
            <summary><span className="tree-chevron">›</span><span className="folder-icon" />{name}</summary>
            <FileTreeLevel node={child} activePath={activePath} previousPaths={previousPaths} onSelect={onSelect} />
          </details>
        </li>
      ))}
      {node.files.map((item) => (
        <li key={item.path}>
          <button
            className={`tree-file ${item.path === activePath ? "active" : ""} ${!previousPaths.has(item.path) ? "new" : ""}`}
            onClick={() => onSelect(item.path)}
            title={item.path}
          >
            <span className={item.content ? "file-icon ready" : "file-icon created"}>PY</span>
            <span>{item.name}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}

function highlightedPythonLines(source) {
  const lines = [[]];

  function appendText(text, classes) {
    text.split("\n").forEach((part, index) => {
      if (index > 0) lines.push([]);
      if (part) lines.at(-1).push({ text: part, classes });
    });
  }

  function visit(value, inheritedClasses = []) {
    if (typeof value === "string") {
      appendText(value, inheritedClasses);
      return;
    }
    if (Array.isArray(value)) {
      value.forEach((item) => visit(item, inheritedClasses));
      return;
    }

    const aliases = Array.isArray(value.alias)
      ? value.alias
      : value.alias ? [value.alias] : [];
    visit(value.content, [...inheritedClasses, "token", value.type, ...aliases]);
  }

  visit(Prism.tokenize(source, Prism.languages.python));
  return lines;
}

function RepositoryView({ checkpoint, previousCheckpoint, checkpointIndex, checkpointCount }) {
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

  const files = checkpoint.repository_snapshot.files;
  const file = files.find((item) => item.path === filePath) ?? files[0];
  const previousFile = previousCheckpoint?.repository_snapshot.files.find(
    (item) => item.path === file?.path,
  );
  const previousPaths = new Set(
    previousCheckpoint?.repository_snapshot.files.map((item) => item.path) ?? [],
  );
  const fileTree = buildFileTree(files);
  const lines = file?.content
    ? (file.content.endsWith("\n") ? file.content.slice(0, -1) : file.content).split("\n")
    : [];
  const highlightedLines = file?.content
    ? highlightedPythonLines(file.content.endsWith("\n") ? file.content.slice(0, -1) : file.content)
    : [];
  const addedLines = new Set(addedLineIndexes(previousFile?.content ?? "", file?.content ?? ""));

  return (
    <div className="repo-view">
      <nav className="file-tree" aria-label="累计 checkpoint 文件树">
        <div className="file-tree-header"><span>EXPLORER</span><small>{files.length} FILES</small></div>
        <div className="file-tree-body">
          {files.length === 0
            ? <span className="empty-tree">EMPTY REPOSITORY</span>
            : <FileTreeLevel node={fileTree} activePath={file?.path} previousPaths={previousPaths} onSelect={setFilePath} />}
        </div>
      </nav>
      <div className="code-view">
        <div className="code-header">
          <span>{file?.path ?? "NO FILE SELECTED"}</span>
          <div>
            {file?.path === checkpoint.active_file && <Pill tone="token">FOCUS</Pill>}
            <span className="line-count">{lines.length} LINES</span>
          </div>
        </div>
        <pre className={`code-panel ${lines.length === 0 ? "empty" : ""}`} ref={codePanelRef} tabIndex="0" aria-label={file ? `${file.path} 源码` : "空仓库"} aria-live="polite">
          {lines.length === 0 ? (
            <span className="empty-file">
              <i>{file ? "EMPTY FILE" : "EMPTY REPOSITORY"}</i>
              <strong>{file ? (checkpoint.step === "step00" ? "本章只建立模块边界，代码从后续章节开始出现。" : "继续阅读，代码会在这里逐步出现。") : "向下阅读，推理模块会在这里逐个建立。"}</strong>
              <small>Checkpoint {String(checkpointIndex + 1).padStart(2, "0")} / {String(checkpointCount).padStart(2, "0")}</small>
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
                    <i>{String(number).padStart(3, "0")}</i>
                    {highlightedLines[index]?.length
                      ? highlightedLines[index].map((segment, segmentIndex) => (
                        segment.classes.length > 0
                          ? <span className={segment.classes.join(" ")} key={`${segmentIndex}-${segment.text}`}>{segment.text}</span>
                          : segment.text
                      ))
                      : " "}
                  </span>
                );
              })}
            </code>
          )}
        </pre>
      </div>
    </div>
  );
}

function Lab({ checkpoint, previousCheckpoint, checkpointIndex, checkpoints, labLabel, pinned, togglePinned, moveCheckpoint, mobileClose }) {
  return (
    <aside className="lab" aria-label={labLabel}>
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
        <RepositoryView checkpoint={checkpoint} previousCheckpoint={previousCheckpoint} checkpointIndex={checkpointIndex} checkpointCount={checkpoints.length} />
      </div>
      <footer className="lab-footer"><span>SOURCE CHECKPOINT</span><span>←/→ STEP · ESC UNLOCK</span><span>INSERT-ONLY SNAPSHOTS</span></footer>
    </aside>
  );
}

function LessonNavigation({ previousStep, nextStep }) {
  return (
    <nav className="lesson-navigation" aria-label="章节导航">
      {previousStep && (
        <a className="lesson-navigation-link previous" href={`${baseUrl}${previousStep.id}/`}>
          <span>← 上一章</span>
          <strong>Step {stepPresentation[previousStep.id].number} · {stepPresentation[previousStep.id].title}</strong>
          <small>{stepPresentation[previousStep.id].summary}</small>
        </a>
      )}
      {nextStep && (
        <a className="lesson-navigation-link next" href={`${baseUrl}${nextStep.id}/`}>
          <span>下一章 →</span>
          <strong>Step {stepPresentation[nextStep.id].number} · {stepPresentation[nextStep.id].title}</strong>
          <small>{stepPresentation[nextStep.id].summary}</small>
        </a>
      )}
    </nav>
  );
}

export function App() {
  const relativePath = window.location.pathname.startsWith(baseUrl)
    ? window.location.pathname.slice(baseUrl.length)
    : window.location.pathname.replace(/^\/+/, "");
  const requestedStep = relativePath.split("/").filter(Boolean)[0] ?? "step00";
  const step = content.steps.find((item) => item.id === requestedStep) ?? content.steps[0];
  const stepIndex = content.steps.findIndex((item) => item.id === step.id);
  const previousStep = content.steps[stepIndex - 1];
  const nextStep = content.steps[stepIndex + 1];
  const presentation = stepPresentation[step.id];
  const checkpoints = step.checkpoints;
  const readingCheckpointIndex = useReadingCheckpoint(checkpoints);
  const [pinnedCheckpointIndex, setPinnedCheckpointIndex] = useState(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [courseNavOpen, setCourseNavOpen] = useState(false);
  const [theme, setTheme] = useState(initialTheme);
  const courseNavRef = useRef(null);
  const checkpointIndex = pinnedCheckpointIndex ?? readingCheckpointIndex;
  const checkpoint = checkpoints[checkpointIndex];
  const previousCheckpoint = checkpointIndex > 0 ? checkpoints[checkpointIndex - 1] : null;

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    window.localStorage.setItem("qwen3-moe-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!courseNavOpen) return undefined;

    const closeCourseNav = (event) => {
      if (event.key === "Escape" || !courseNavRef.current?.contains(event.target)) {
        setCourseNavOpen(false);
      }
    };

    document.addEventListener("pointerdown", closeCourseNav);
    document.addEventListener("keydown", closeCourseNav);
    return () => {
      document.removeEventListener("pointerdown", closeCourseNav);
      document.removeEventListener("keydown", closeCourseNav);
    };
  }, [courseNavOpen]);

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
    checkpoints,
    labLabel: presentation.labLabel,
    pinned: pinnedCheckpointIndex !== null,
    togglePinned,
    moveCheckpoint,
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href={baseUrl}><span>Q3</span><strong>MOE INFERENCE LAB</strong></a>
        <div className="course-nav" ref={courseNavRef}>
          <button className="course-progress" onClick={() => setCourseNavOpen((open) => !open)} aria-expanded={courseNavOpen} aria-haspopup="true">
            <span>STEP {presentation.number} / 14</span>
            <div><i style={{ width: `${presentation.progress}%` }} /></div>
            <b>{presentation.label}</b>
            <small aria-hidden="true">⌄</small>
          </button>
          {courseNavOpen && (
            <nav className="course-menu" aria-label="课程 Step 导航">
              <header><span>COURSE STEPS</span><small>选择章节</small></header>
              {content.steps.map((courseStep) => {
                const item = stepPresentation[courseStep.id];
                const current = courseStep.id === step.id;
                return (
                  <a className={current ? "current" : ""} href={`${baseUrl}${courseStep.id}/`} aria-current={current ? "page" : undefined} key={courseStep.id}>
                    <span>{item.number}</span>
                    <div><strong>{item.title}</strong><small>{item.summary}</small></div>
                    <b>{current ? "CURRENT" : "OPEN"}</b>
                  </a>
                );
              })}
              <div className="course-next"><span>NEXT</span><strong>STEP 08 · CAUSAL LM PREFILL</strong><small>COMING SOON</small></div>
            </nav>
          )}
        </div>
        <div className="topbar-actions">
          <button
            className="theme-button"
            onClick={() => setTheme((current) => current === "dark" ? "light" : "dark")}
            aria-label={`切换到${theme === "dark" ? "浅色" : "深色"}模式`}
            title={`切换到${theme === "dark" ? "浅色" : "深色"}模式`}
          >
            <span aria-hidden="true">{theme === "dark" ? "☀" : "◐"}</span>
            {theme === "dark" ? "LIGHT" : "DARK"}
          </button>
          <a className="source-button" href="https://github.com/CypressVillage/learn-qwen3-moe" aria-label="查看仓库源码">SOURCE ↗</a>
        </div>
      </header>
      <main className="workspace">
        <Lab {...labProps} />
        <article className="lesson-pane">
          <div className="lesson-kicker"><Pill tone="token">{presentation.kicker}</Pill><span>{presentation.duration}</span></div>
          <div className="lesson-content" dangerouslySetInnerHTML={{ __html: step.lesson.html }} />
          <LessonNavigation previousStep={previousStep} nextStep={nextStep} />
        </article>
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
