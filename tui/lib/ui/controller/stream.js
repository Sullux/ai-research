const { refs, formatTimestamp } = require('./state')

const getItemColors = (type) => {
  if (type === 'thought') return { titleFg: '#bb9af7', contentFg: '#7a88cf', bg: '#14141e' }
  if (type === 'tool_call') return { titleFg: '#e0af68', contentFg: '#ff9e64', bg: '#1f1a16' }
  if (type === 'tool_result') return { titleFg: '#9ece6a', contentFg: '#73daca', bg: '#141e18' }
  if (type === 'user') return { titleFg: '#7aa2f7', contentFg: '#e2e8f0', bg: '#162b4d' }
  return { titleFg: '#9aa5ce', contentFg: '#c0caf5', bg: '#151521' }
}

const liveNode = (title, text, fg, bg) => ({
  type: 'layout',
  direction: 'vertical',
  bg,
  margin: { top: 0, bottom: 1 },
  padding: { top: 0, bottom: 0, left: 1, right: 1 },
  inner: [{
    type: 'rich',
    inner: [
      { type: 'text', text: '▶ ', bold: true, fg },
      { type: 'text', text: `[${formatTimestamp(Date.now())}] `, fg: '#565f89' },
      { type: 'text', text: `${title}: `, bold: true, fg },
      { type: 'text', text: text, fg },
      { type: 'text', text: ' ▍', fg, bold: true },
    ],
  }],
})

const getStreamNodes = () => {
  if (!refs.store) return []
  const nodes = refs.store.state.stream.map((item, idx) => {
    const isSelected = refs.store.state.mode === 'stream' && refs.store.state.selectedIdx.stream === idx
    const { titleFg, contentFg, bg } = getItemColors(item.type)
    const preview = item.content.length > 80 && !item.expanded
      ? `${item.content.slice(0, 77)}... (press x to expand)`
      : item.content

    return {
      type: 'layout',
      direction: 'vertical',
      bg,
      margin: { top: 0, bottom: 1 },
      padding: { top: 0, bottom: 0, left: 1, right: 1 },
      inner: [{
        type: 'rich',
        inner: [
          { type: 'text', text: isSelected ? '▶ ' : '  ', bold: true, fg: '#f7768e' },
          { type: 'text', text: `[${formatTimestamp(item.time)}] `, fg: '#565f89' },
          { type: 'text', text: `${item.title}: `, bold: true, fg: titleFg },
          { type: 'text', text: preview, fg: contentFg },
        ],
      }],
    }
  })

  if (refs.store.state.activeThought) {
    nodes.push(liveNode('💭 THOUGHT', refs.store.state.activeThought, '#bb9af7', '#14141e'))
  }
  if (refs.store.state.activeResponse) {
    nodes.push(liveNode('🤖 ASSISTANT', refs.store.state.activeResponse, '#7dcfff', '#132133'))
  }
  return nodes
}

module.exports = { getStreamNodes }
