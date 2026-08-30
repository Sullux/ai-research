const { refs, formatTimestamp } = require('./state')

const getStreamNodes = () => {
  if (!refs.store) return []
  const list = refs.store.state.stream
  return list.map((item, idx) => {
    const isSelected =
      refs.store.state.mode === 'stream' &&
      refs.store.state.selectedIdx.stream === idx
    const timeStr = formatTimestamp(item.time)

    let titleFg = '#9aa5ce'
    let contentFg = '#c0caf5'
    let bg = '#151521'

    if (item.type === 'thought') {
      titleFg = '#bb9af7'
      contentFg = '#7a88cf'
      bg = '#14141e'
    } else if (item.type === 'tool_call') {
      titleFg = '#e0af68'
      contentFg = '#ff9e64'
      bg = '#1f1a16'
    } else if (item.type === 'tool_result') {
      titleFg = '#9ece6a'
      contentFg = '#73daca'
      bg = '#141e18'
    } else if (item.type === 'user') {
      titleFg = '#7aa2f7'
      contentFg = '#e2e8f0'
      bg = '#162b4d'
    }

    const preview =
      item.content.length > 80 && !item.expanded
        ? `${item.content.slice(0, 77)}... (press x to expand)`
        : item.content

    return {
      type: 'layout',
      direction: 'vertical',
      bg,
      margin: { top: 0, bottom: 1 },
      padding: { top: 0, bottom: 0, left: 1, right: 1 },
      inner: [
        {
          type: 'rich',
          inner: [
            {
              type: 'text',
              text: isSelected ? '▶ ' : '  ',
              bold: true,
              fg: '#f7768e',
            },
            { type: 'text', text: `[${timeStr}] `, fg: '#565f89' },
            { type: 'text', text: `${item.title}: `, bold: true, fg: titleFg },
            { type: 'text', text: preview, fg: contentFg },
          ],
        },
      ],
    }
  })
}

module.exports = {
  getStreamNodes,
}
