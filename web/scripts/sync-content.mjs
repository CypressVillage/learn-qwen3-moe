import { readFile, mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { marked } from "marked";


const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
const outputPath = resolve(here, "../src/generated/content.json");
const stepSources = [
  {
    id: "step00",
    lessonPath: "lessons/step00-inference-map.md",
    checkpointPath: "lessons/checkpoints/step00.json",
  },
  {
    id: "step01",
    lessonPath: "lessons/step01-overview-config-weights.md",
    checkpointPath: "lessons/checkpoints/step01.json",
  },
  {
    id: "step02",
    lessonPath: "lessons/step02-tokenizer.md",
    checkpointPath: "lessons/checkpoints/step02.json",
  },
  {
    id: "step03",
    lessonPath: "lessons/step03-basic-layers.md",
    checkpointPath: "lessons/checkpoints/step03.json",
  },
  {
    id: "step04",
    lessonPath: "lessons/step04-rope.md",
    checkpointPath: "lessons/checkpoints/step04.json",
  },
  {
    id: "step05",
    lessonPath: "lessons/step05-gqa-attention.md",
    checkpointPath: "lessons/checkpoints/step05.json",
  },
  {
    id: "step06",
    lessonPath: "lessons/step06-sparse-moe.md",
    checkpointPath: "lessons/checkpoints/step06.json",
  },
  {
    id: "step07",
    lessonPath: "lessons/step07-decoder-layer.md",
    checkpointPath: "lessons/checkpoints/step07.json",
  },
  {
    id: "step08",
    lessonPath: "lessons/step08-causal-lm-prefill.md",
    checkpointPath: "lessons/checkpoints/step08.json",
  },
  {
    id: "step09",
    lessonPath: "lessons/step09-next-token-selection.md",
    checkpointPath: "lessons/checkpoints/step09.json",
  },
  {
    id: "step10",
    lessonPath: "lessons/step10-kv-cache.md",
    checkpointPath: "lessons/checkpoints/step10.json",
  },
  {
    id: "step11",
    lessonPath: "lessons/step11-cached-attention.md",
    checkpointPath: "lessons/checkpoints/step11.json",
  },
  {
    id: "step12",
    lessonPath: "lessons/step12-cached-decode.md",
    checkpointPath: "lessons/checkpoints/step12.json",
  },
];

marked.use({
  gfm: true,
  breaks: false,
  renderer: {
    heading({ tokens, depth }) {
      const text = this.parser.parseInline(tokens);
      const plain = text.replace(/<[^>]+>/g, "");
      const id = plain
        .toLowerCase()
        .replace(/[^a-z0-9\u4e00-\u9fff]+/g, "-")
        .replace(/(^-|-$)/g, "");
      return `<h${depth} id="${id}">${text}</h${depth}>`;
    },
  },
});

const steps = await Promise.all(stepSources.map(async (source) => {
  const [markdown, checkpointText] = await Promise.all([
    readFile(resolve(root, source.lessonPath), "utf8"),
    readFile(resolve(root, source.checkpointPath), "utf8"),
  ]);
  const markdownWithAnchors = markdown.replace(
    /<!-- checkpoint: ([a-z0-9-]+) -->/g,
    '<div class="checkpoint-anchor" data-checkpoint="$1" aria-hidden="true"></div>',
  );
  return {
    id: source.id,
    lesson: {
      path: source.lessonPath,
      markdown,
      html: await marked.parse(markdownWithAnchors),
    },
    checkpoints: JSON.parse(checkpointText).checkpoints,
  };
}));

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify({ steps })}\n`, "utf8");
console.log(`synced ${steps.length} course lessons`);
