const { refs, formatTimestamp } = require('./state')

const getConversationScroll = () => (refs.store?.state.stickyScroll.chat ? Infinity : 0)

const getConversationNodes = () => {
  if (!refs.store) return []
  const { conversation, mode, isEditMode, selectedIdx, activeResponse } = refs.store.state
  const isChatFocused = mode === 'chat' && !isEditMode

  const nodes = conversation.map((msg, idx) => {
    const isUser = msg.sender === 'User'
    const isSelected = isChatFocused && selectedIdx.chat === idx
    const timeStr = formatTimestamp(msg.time)

    let bg = isUser ? '#162b4d' : '#132133'
    let headerFg = isUser ? '#7aa2f7' : '#7dcfff'
    let headerPrefix = isUser ? '👤 ' : '🤖 '

    if (msg.waitingUser) {
      bg = '#2b2314'
      headerFg = '#e0af68'
      headerPrefix = '⏳ '
    }

    return {
      type: 'layout',
      direction: 'vertical',
      bg,
      margin: { top: 0, bottom: 1 },
      padding: isUser ? { top: 0, bottom: 0, left: 2, right: 1 } : { top: 0, bottom: 0, left: 1, right: 2 },
      inner: [{
        type: 'rich',
        inner: [
          { type: 'text', text: isSelected ? '▶ ' : '  ', bold: true, fg: '#f7768e' },
          { type: 'text', text: `[${timeStr}] `, fg: '#565f89' },
          { type: 'text', text: `${headerPrefix}${msg.sender}: `, bold: true, fg: headerFg },
          { type: 'text', text: msg.text, fg: isUser ? '#e2e8f0' : '#c0caf5' },
        ],
      }],
    }
  })

  if (activeResponse) {
    const timeStr = formatTimestamp(Date.now())
    nodes.push({
      type: 'layout',
      direction: 'vertical',
      bg: '#132133',
      margin: { top: 0, bottom: 1 },
      padding: { top: 0, bottom: 0, left: 1, right: 2 },
      inner: [{
        type: 'rich',
        inner: [
          { type: 'text', text: '  ', fg: '#7dcfff' },
          { type: 'text', text: `[${timeStr}] `, fg: '#565f89' },
          { type: 'text', text: '🤖 Assistant: ', bold: true, fg: '#7dcfff' },
          { type: 'text', text: activeResponse, fg: '#c0caf5' },
          { type: 'text', text: ' ▍', fg: '#7aa2f7', bold: true },
        ],
      }],
    })
  }

  return nodes
}

module.exports = {
  getConversationScroll,
  getConversationNodes,
}
