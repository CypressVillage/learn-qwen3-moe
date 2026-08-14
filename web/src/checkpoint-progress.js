function sourceLines(source) {
  if (!source) return [];
  return source.endsWith("\n") ? source.slice(0, -1).split("\n") : source.split("\n");
}

export function checkpointIndexAtReadingLine(anchorTops, readingLine) {
  let activeIndex = 0;
  anchorTops.forEach((top, index) => {
    if (top <= readingLine) activeIndex = index;
  });
  return activeIndex;
}

export function checkpointIndexAfterKey(key, currentIndex, checkpointCount) {
  if (key !== "ArrowLeft" && key !== "ArrowRight") return null;
  const direction = key === "ArrowLeft" ? -1 : 1;
  return Math.max(0, Math.min(checkpointCount - 1, currentIndex + direction));
}

export function addedLineIndexes(previousSource, currentSource) {
  const previous = sourceLines(previousSource);
  const current = sourceLines(currentSource);
  const lengths = Array.from({ length: previous.length + 1 }, () =>
    Array(current.length + 1).fill(0),
  );

  for (let previousIndex = previous.length - 1; previousIndex >= 0; previousIndex -= 1) {
    for (let currentIndex = current.length - 1; currentIndex >= 0; currentIndex -= 1) {
      lengths[previousIndex][currentIndex] = previous[previousIndex] === current[currentIndex]
        ? lengths[previousIndex + 1][currentIndex + 1] + 1
        : Math.max(
          lengths[previousIndex + 1][currentIndex],
          lengths[previousIndex][currentIndex + 1],
        );
    }
  }

  const preserved = new Set();
  let previousIndex = 0;
  let currentIndex = 0;
  while (previousIndex < previous.length && currentIndex < current.length) {
    if (previous[previousIndex] === current[currentIndex]) {
      preserved.add(currentIndex);
      previousIndex += 1;
      currentIndex += 1;
    } else if (
      lengths[previousIndex + 1][currentIndex]
      >= lengths[previousIndex][currentIndex + 1]
    ) {
      previousIndex += 1;
    } else {
      currentIndex += 1;
    }
  }
  return current.map((_, index) => index).filter((index) => !preserved.has(index));
}
