const { refs, formatTimestamp } = require('./state')
const { getLayoutTier } = require('./layout')

const getStreamScroll = () => (refs.store?.state.stickyScroll.stream ? Infinity : 0)

const getItemColors = (t) => {
  if (t === 'thought') return { titleFg: '#bb9af7', contentFg: '#7a88cf', bg: '#14141e' }
  if (t === 'tool_call') return { titleFg: '#e0af68', contentFg: '#ff9e64', bg: '#1f1a16' }
  if (t === 'tool_result') return { titleFg: '#9ece6a', contentFg: '#73daca', bg: '#141e18' }
  if (t === 'user') return { titleFg: '#7aa2f7', contentFg: '#e2e8f0', bg: '#162b4d' }
  return { titleFg: '#9aa5ce', contentFg: '#c0caf5', bg: '#151521' }
}

const getPanelWidth = () => {
  const cols = refs.store?.state?.dimensions?.cols || 100
  const ratio = getLayoutTier(cols) === 1 ? 0.3 : (getLayoutTier(cols) === 2 ? 0.4 : 0.95)
  return Math.max(30, Math.floor(cols * ratio) - 4)
}

const wrapLines = (text, maxWidth) => {
  if (!text) return []
  const maxW = Math.max(20, maxWidth || 60)
  const lines = []
  for (const para of text.split('\n')) {
    if (!para) { lines.push(''); continue }
    let cur = ''
    for (const w of para.split(/\s+/)) {
      if (!cur) cur = w
      else if (cur.length + 1 + w.length <= maxW) cur += ' ' + w
      else { lines.push(cur); cur = w }
    }
    if (cur) lines.push(cur)
  }
  return lines
}

const buildCardSpans = (item, isSel, width) => {
  const { titleFg, contentFg } = getItemColors(item.type)
  const lines = wrapLines(item.content, width)
  const spans = [
    { type: 'text', text: isSel ? '▶ ' : '  ', bold: true, fg: '#f7768e' },
    { type: 'text', text: `[${formatTimestamp(item.time)}] `, fg: '#565f89' },
    { type: 'text', text: `${item.title}`, bold: true, fg: titleFg },
  ]
  if (lines.length <= 3) {
    spans.push({ type: 'text', text: `: ${lines[0] || ''}`, fg: contentFg })
    for (let i = 1; i < lines.length; i++) spans.push({ type: 'text', text: `\n    ${lines[i]}`, fg: contentFg })
    return spans
  }
  if (!item.expanded) {
    spans.push({ type: 'text', text: `  (↑ ${lines.length - 3} more)`, fg: '#565f89', italic: true })
    if (isSel) spans.push({ type: 'text', text: '  (x to expand)', fg: '#e0af68', bold: true })
    for (const l of lines.slice(-3)) spans.push({ type: 'text', text: `\n  ${l}`, fg: contentFg })
  } else {
    if (isSel) spans.push({ type: 'text', text: '  (x to collapse)', fg: '#e0af68', bold: true })
    for (const l of lines) spans.push({ type: 'text', text: `\n  ${l}`, fg: contentFg })
  }
  return spans
}

const liveNode = (title, text, fg, bg, width) => ({
  type: 'layout', direction: 'vertical', bg, margin: { top: 0, bottom: 1 }, padding: { top: 0, bottom: 0, left: 1, right: 1 },
  inner: [{
    type: 'rich',
    inner: [
      { type: 'text', text: '  ', fg },
      { type: 'text', text: `[${formatTimestamp(Date.now())}] `, fg: '#565f89' },
      { type: 'text', text: `${title}`, bold: true, fg },
      { type: 'text', text: wrapLines(text, width).slice(-3).map((l) => `\n  ${l}`).join(''), fg },
      { type: 'text', text: ' ▍', fg, bold: true },
    ],
  }],
})

const cleanThought = (raw) => (raw || '').trim().replace(/^thought\s*/, '')

const getStreamNodes = () => {
  if (!refs.store) return []
  const width = getPanelWidth()
  const selIdx = refs.store.state.mode === 'stream' ? refs.store.state.selectedIdx.stream : -1
  const nodes = refs.store.state.stream.map((item, idx) => ({
    type: 'layout', direction: 'vertical', bg: getItemColors(item.type).bg, margin: { top: 0, bottom: 1 }, padding: { top: 0, bottom: 0, left: 1, right: 1 },
    inner: [{ type: 'rich', inner: buildCardSpans(item, idx === selIdx, width) }],
  }))
  const th = cleanThought(refs.store.state.activeThought)
  if (th) nodes.push(liveNode('💭 THOUGHT', th, '#bb9af7', '#14141e', width))
  if (refs.store.state.activeResponse) nodes.push(liveNode('🤖 ASSISTANT', refs.store.state.activeResponse, '#7dcfff', '#132133', width))
  return nodes
}

module.exports = { getStreamScroll, getStreamNodes }
