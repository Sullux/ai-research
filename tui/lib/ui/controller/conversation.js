const { refs, formatTimestamp } = require('./state')

const getConversationNodes = () => {
  if (!refs.store) return []
  const list = refs.store.state.conversation
  return list.map((msg, idx) => {
    const isUser = msg.sender === 'User'
    const isSelected =
      refs.store.state.mode === 'chat' && refs.store.state.selectedIdx.chat === idx
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
      padding: isUser
        ? { top: 0, bottom: 0, left: 2, right: 1 }
        : { top: 0, bottom: 0, left: 1, right: 2 },
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
            {
              type: 'text',
              text: `${headerPrefix}${msg.sender}: `,
              bold: true,
              fg: headerFg,
            },
            {
              type: 'text',
              text: msg.text,
              fg: isUser ? '#e2e8f0' : '#c0caf5',
            },
          ],
        },
      ],
    }
  })
}

module.exports = {
  getConversationNodes,
}
