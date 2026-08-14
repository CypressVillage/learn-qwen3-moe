import { readFile, mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { marked } from "marked";


const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
const lessonPath = resolve(root, "lessons/step01-overview-config-weights.md");
const checkpointPath = resolve(root, "lessons/checkpoints/step01.json");
const outputPath = resolve(here, "../src/generated/content.json");

const [markdown, checkpointText] = await Promise.all([
  readFile(lessonPath, "utf8"),
  readFile(checkpointPath, "utf8"),
]);

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

const markdownWithAnchors = markdown
  .replace(
    /<!-- checkpoint: ([a-z0-9-]+) -->/g,
    '<div class="checkpoint-anchor" data-checkpoint="$1" aria-hidden="true"></div>',
  );
const data = {
  lesson: {
    path: "lessons/step01-overview-config-weights.md",
    markdown,
    html: await marked.parse(markdownWithAnchors),
  },
  checkpoints: JSON.parse(checkpointText).checkpoints,
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(data)}\n`, "utf8");
console.log("synced Step 01 lesson and checkpoints");
