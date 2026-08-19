import { readFile, mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { marked } from "marked";


const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
const outputPath = resolve(here, "../src/generated/content.json");
const glossaryPath = resolve(root, "lessons/glossary.json");
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
  {
    id: "step13",
    lessonPath: "lessons/step13-autoregressive-generation.md",
    checkpointPath: "lessons/checkpoints/step13.json",
  },
  {
    id: "step14",
    lessonPath: "lessons/step14-end-to-end-inference.md",
    checkpointPath: "lessons/checkpoints/step14.json",
  },
];

const glossary = JSON.parse(await readFile(glossaryPath, "utf8"));

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function updateFence(line, fence) {
  const match = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
  if (!match) return { fence, changed: false };
  const marker = match[1];
  const trailing = match[2];
  if (!fence) {
    if (marker[0] === "`" && trailing.includes("`")) return { fence, changed: false };
    return { fence: marker, changed: true };
  }
  if (marker[0] === fence[0] && marker.length >= fence.length && !trailing.replaceAll(" ", "")) {
    return { fence: null, changed: true };
  }
  return { fence, changed: false };
}

function validatePrincipleBlocks(markdown, lessonPath) {
  let active = null;
  let bodyHasContent = false;
  let fence = null;
  markdown.split("\n").forEach((line, index) => {
    const lineNumber = index + 1;
    const transition = updateFence(line, fence);
    if (transition.changed) {
      fence = transition.fence;
      if (active) bodyHasContent = true;
      return;
    }
    if (fence) {
      if (active && line.trim()) bodyHasContent = true;
      return;
    }

    const opener = /^:::principle[ \t]+(\S.*)$/.exec(line);
    if (opener) {
      if (active) throw new Error(`nested principle block: ${lessonPath}:${lineNumber}`);
      active = { title: opener[1], lineNumber };
      bodyHasContent = false;
      return;
    }
    if (/^:::endprinciple[ \t]*$/.test(line)) {
      if (!active) throw new Error(`orphan principle closer: ${lessonPath}:${lineNumber}`);
      if (!bodyHasContent) {
        throw new Error(`empty principle block: ${lessonPath}:${active.lineNumber}`);
      }
      active = null;
      return;
    }
    if (line.trimStart().startsWith(":::principle") || line.trimStart().startsWith(":::endprinciple")) {
      throw new Error(`malformed principle delimiter: ${lessonPath}:${lineNumber}`);
    }
    if (active) {
      if (line.includes("<!-- checkpoint:")) {
        throw new Error(`checkpoint inside principle block: ${lessonPath}:${lineNumber}`);
      }
      if (line.trim()) bodyHasContent = true;
    }
  });
  if (active) throw new Error(`unterminated principle block: ${lessonPath}:${active.lineNumber}`);
}

function principleBlock(source, lexer) {
  const opener = /^:::principle[ \t]+([^\n]+)\n/.exec(source);
  if (!opener) return undefined;

  let offset = opener[0].length;
  let fence = null;
  while (offset <= source.length) {
    const lineEnd = source.indexOf("\n", offset);
    const nextOffset = lineEnd === -1 ? source.length : lineEnd + 1;
    const line = source.slice(offset, lineEnd === -1 ? source.length : lineEnd);
    const transition = updateFence(line, fence);
    if (transition.changed) {
      fence = transition.fence;
    } else if (!fence && /^:::endprinciple[ \t]*$/.test(line)) {
      const body = source.slice(opener[0].length, offset).trim();
      if (!body) throw new Error(`empty principle block: ${opener[1].trim()}`);
      if (/<!-- checkpoint:/.test(body)) {
        throw new Error(`checkpoint inside principle block: ${opener[1].trim()}`);
      }
      return {
        type: "principleBlock",
        raw: source.slice(0, nextOffset),
        title: opener[1].trim(),
        tokens: lexer.blockTokens(body),
      };
    }
    if (lineEnd === -1) break;
    offset = nextOffset;
  }
  throw new Error(`unterminated principle block: ${opener[1].trim()}`);
}

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
  extensions: [
    {
      name: "principleBlock",
      level: "block",
      start(source) {
        const match = /^:::principle[ \t]+\S/m.exec(source);
        return match?.index;
      },
      tokenizer(source) {
        return principleBlock(source, this.lexer);
      },
      renderer(token) {
        const title = escapeHtml(token.title);
        return `<details class="principle-block"><summary class="principle-summary"><span class="principle-label">原理深入</span><span class="principle-title">${title}</span><span class="principle-chevron" aria-hidden="true">&rsaquo;</span></summary><div class="principle-body">${this.parser.parse(token.tokens)}</div></details>\n`;
      },
      childTokens: ["tokens"],
    },
    {
      name: "glossaryTerm",
      level: "inline",
      start(source) {
        const index = source.indexOf("[[");
        return index === -1 ? undefined : index;
      },
      tokenizer(source) {
        const match = /^\[\[([^\]\n]+)\]\]/.exec(source);
        if (!match) return undefined;
        return { type: "glossaryTerm", raw: match[0], key: match[1].trim() };
      },
      renderer(token) {
        if (!Object.hasOwn(glossary, token.key)) {
          throw new Error(`unknown glossary term: ${token.key}`);
        }
        const key = escapeHtml(token.key);
        return `<button class="glossary-term" type="button" data-glossary-key="${key}" aria-label="${key}：查看术语解释">${key}</button>`;
      },
    },
  ],
});

const steps = await Promise.all(stepSources.map(async (source) => {
  const [markdown, checkpointText] = await Promise.all([
    readFile(resolve(root, source.lessonPath), "utf8"),
    readFile(resolve(root, source.checkpointPath), "utf8"),
  ]);
  validatePrincipleBlocks(markdown, source.lessonPath);
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
await writeFile(outputPath, `${JSON.stringify({ glossary, steps })}\n`, "utf8");
console.log(`synced ${steps.length} course lessons`);
