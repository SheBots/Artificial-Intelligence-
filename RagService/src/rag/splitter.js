// Simple text splitter that aims for target token/char size with overlap
export function chunkText(text, opts = {}) {
  const target = opts.target || 700; // approximate chars
  const overlap = opts.overlap || 120;
  const paragraphs = text.split(/\n{2,}|\r\n{2,}/).map(p => p.trim()).filter(Boolean);

  const chunks = [];
  let current = '';

  for (const p of paragraphs) {
    if ((current + '\n\n' + p).length <= target) {
      current = current ? current + '\n\n' + p : p;
    } else {
      if (current) chunks.push(current.trim());
      if (p.length <= target) {
        current = p;
      } else {
        // split long paragraph by sentences
        const sentences = p.split(/(?<=[.!?])\s+/);
        let buf = '';
        for (const s of sentences) {
          if ((buf + ' ' + s).trim().length <= target) {
            buf = buf ? buf + ' ' + s : s;
          } else {
            if (buf) chunks.push(buf.trim());
            buf = s;
          }
        }
        if (buf) current = buf;
        else current = '';
      }
    }
    // apply overlap if needed
    if (current.length > target) {
      chunks.push(current.trim());
      current = current.slice(-overlap);
    }
  }

  if (current) chunks.push(current.trim());

  // Merge very small chunks
  const merged = [];
  for (const c of chunks) {
    if (merged.length === 0) merged.push(c);
    else if (merged[merged.length - 1].length < Math.floor(target / 4)) merged[merged.length - 1] += '\n\n' + c;
    else merged.push(c);
  }

  return merged;
}
