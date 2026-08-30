const formatTimestamp = (ts) => {
  const d = new Date(ts)
  const pad = (n) => String(n).padStart(2, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

let store = null
let client = null
let session = null
let orchestrator = null
let timers = null
let lastLayoutTier = 1

const init = (s, c, sess, orch, tmrs) => {
  store = s
  client = c
  session = sess
  orchestrator = orch
  timers = tmrs
}

const getLayoutTier = (cols) => {
  const width = cols || store?.state?.dimensions?.cols || 100
  if (width >= 200) return 1
  if (width >= 160) return 2
  return 3
}

const isClipped = (cols, rows) => {
  const c = cols || store?.state?.dimensions?.cols || 100
  const r = rows || store?.state?.dimensions?.rows || 30
  return c < 80 || r < 30
}

const getClippedBanner = () => {
  return ' ⚠️  TERMINAL TOO SMALL (Minimum 80x30 required) — CONTENT CLIPPED '
}

const getStatusText = () => {
  if (!store) return 'Status: Ready'
  const state = store.state
  const pausedTag = state.isPaused ? ' [⏸️ PAUSED]' : ''
  return ` ${state.status}${pausedTag}`
}

const getShortcutsText = () => {
  if (!store) return ' [Enter] Type  [a] Chat  [s] Stream  [d] Plan  [Ctrl+Q] Quit'
  const state = store.state
  if (state.isPaused) {
    return ' [r] Resume Inference | [Ctrl+Q] Quit'
  }
  if (state.isEditMode) {
    return ' [Enter] Send | [Esc] Normal Mode | [Shift+Enter] Newline | [Ctrl+C] Clear'
  }
  if (state.mode === 'chat') {
    return ' [Enter] Type | [j/k] Scroll | [s] Stream | [d] Plan | [c] Copy | [Esc] Pause | [Ctrl+Q] Quit'
  }
  if (state.mode === 'stream') {
    return ' [j/k] Scroll | [x] Expand/Collapse | [a] Chat | [d] Plan | [c] Copy | [Esc] Normal/Pause'
  }
  if (state.mode === 'plan') {
    return ' [j/k] Navigate Tasks | [a] Chat | [s] Stream | [c] Copy | [Esc] Normal/Pause'
  }
  return ' [Enter] Type | [a] Chat | [s] Stream | [d] Plan | [Ctrl+Q] Quit'
}

const getConversationNodes = () => {
  if (!store) return []
  const list = store.state.conversation
  return list.map((msg, idx) => {
    const isUser = msg.sender === 'User'
    const isSelected = store.state.mode === 'chat' && store.state.selectedIdx.chat === idx
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
            { type: 'text', text: isSelected ? '▶ ' : '  ', bold: true, fg: '#f7768e' },
            { type: 'text', text: `[${timeStr}] `, fg: '#565f89' },
            { type: 'text', text: `${headerPrefix}${msg.sender}: `, bold: true, fg: headerFg },
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

const getStreamNodes = () => {
  if (!store) return []
  const list = store.state.stream
  return list.map((item, idx) => {
    const isSelected = store.state.mode === 'stream' && store.state.selectedIdx.stream === idx
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

    const preview = item.content.length > 80 && !item.expanded
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
            { type: 'text', text: isSelected ? '▶ ' : '  ', bold: true, fg: '#f7768e' },
            { type: 'text', text: `[${timeStr}] `, fg: '#565f89' },
            { type: 'text', text: `${item.title}: `, bold: true, fg: titleFg },
            { type: 'text', text: preview, fg: contentFg },
          ],
        },
      ],
    }
  })
}

const getPlanNodes = () => {
  if (!orchestrator) return []
  const plans = orchestrator.plans
  if (plans.length === 0) {
    return [
      {
        type: 'text',
        text: '  (No active plans. Model idle.)',
        fg: '#565f89',
        italic: true,
      },
    ]
  }

  const nodes = []
  for (const p of plans) {
    const isPlanDone = p.status === 'DONE'
    const planIcon = isPlanDone ? '✅ ' : '📋 '
    nodes.push({
      type: 'text',
      text: `${planIcon}Plan ${p.id}: ${p.brief}\n`,
      bold: true,
      fg: isPlanDone ? '#9ece6a' : '#7aa2f7',
      margin: { top: 1, bottom: 0 },
    })

    for (const s of p.steps) {
      let icon = '⬜ '
      let fg = '#9aa5ce'
      let tag = ''

      if (s.status === 'DONE') {
        icon = '✅ '
        fg = '#73daca'
      } else if (s.status === 'IN_PROGRESS') {
        icon = '▶ '
        fg = '#e0af68'
        tag = ' [ACTIVE]'
      } else if (s.status === 'WAITING_FOR_USER') {
        icon = '⏳ '
        fg = '#f7768e'
        tag = ' [WAITING]'
      } else if (s.status === 'DEFERRED') {
        icon = '⏱️ '
        fg = '#bb9af7'
        tag = ` [${s.deferReason || 'timer'}]`
      }

      nodes.push({
        type: 'text',
        text: `   ${icon}${s.id}: ${s.brief}${tag}\n`,
        fg,
      })
    }
  }
  return nodes
}

const onSubmitInput = (ctx, payload) => {
  const val = payload.value?.trim()
  if (payload.node) {
    payload.node.value = ''
    payload.node.cursor = 0
  }
  if (!val || !client) return

  store?.pushHistory(val)
  store?.addConversationMessage({ sender: 'User', text: val })
  store?.addStreamEntry({ type: 'user', title: '👤 USER', content: val })

  // Build appropriate thought prefix
  const thoughtPrefix = orchestrator
    ? orchestrator.buildThoughtPrefix('USER_PROMPT', { message: val })
    : `<|turn>user\n${val}\n<turn|>\n<|turn>model\n<|think|>\n`

  store?.setGenerating(true)
  client.sendInput(thoughtPrefix)
  ctx.redraw()
}

const onGlobalKey = (ctx, event) => {
  if (event.key === 'ctrl+q') {
    process.exit(0)
  }

  // If in Edit Mode, Esc drops back to normal mode
  if (store?.state.isEditMode) {
    if (event.key === 'escape') {
      event.stopPropagation()
      store.setEditMode(false)
      ctx.setFocus?.(null)
      ctx.redraw()
      return
    }
    return
  }

  // Normal / Browse Mode Keybindings
  if (event.key === 'a') {
    event.stopPropagation()
    store?.setMode('chat')
    store?.setOverlay(null)
    ctx.redraw()
    return
  }

  if (event.key === 's') {
    event.stopPropagation()
    const tier = getLayoutTier(ctx.width)
    store?.setMode('stream')
    if (tier >= 3) {
      store?.setOverlay(store.state.overlay === 'stream' ? null : 'stream')
    }
    ctx.redraw()
    return
  }

  if (event.key === 'd') {
    event.stopPropagation()
    const tier = getLayoutTier(ctx.width)
    store?.setMode('plan')
    if (tier >= 2) {
      store?.setOverlay(store.state.overlay === 'plan' ? null : 'plan')
    }
    ctx.redraw()
    return
  }

  if (event.key === 'enter') {
    event.stopPropagation()
    store?.setEditMode(true)
    store?.setMode('chat')
    ctx.setFocus?.('chatInput')
    ctx.redraw()
    return
  }

  if (event.key === 'escape') {
    event.stopPropagation()
    if (store?.state.overlay) {
      store.setOverlay(null)
    } else {
      const p = !store?.state.isPaused
      store?.setPaused(p)
      if (p) client?.sendAbort?.()
    }
    ctx.redraw()
    return
  }

  if (event.key === 'r' && store?.state.isPaused) {
    event.stopPropagation()
    store.setPaused(false)
    ctx.redraw()
    return
  }

  if (event.key === 'x' && store?.state.mode === 'stream') {
    event.stopPropagation()
    store.toggleExpandStreamItem(store.state.selectedIdx.stream)
    ctx.redraw()
    return
  }

  // Scroll navigation j / k
  if (event.key === 'j' || event.key === 'down') {
    event.stopPropagation()
    const mode = store?.state.mode
    if (mode === 'chat') {
      const len = store.state.conversation.length
      if (store.state.selectedIdx.chat < len - 1) {
        store.state.selectedIdx.chat += 1
      }
    } else if (mode === 'stream') {
      const len = store.state.stream.length
      if (store.state.selectedIdx.stream < len - 1) {
        store.state.selectedIdx.stream += 1
      }
    }
    ctx.redraw()
    return
  }

  if (event.key === 'k' || event.key === 'up') {
    event.stopPropagation()
    const mode = store?.state.mode
    if (mode === 'chat') {
      if (store.state.selectedIdx.chat > 0) {
        store.state.selectedIdx.chat -= 1
        store.state.stickyScroll.chat = false
      }
    } else if (mode === 'stream') {
      if (store.state.selectedIdx.stream > 0) {
        store.state.selectedIdx.stream -= 1
        store.state.stickyScroll.stream = false
      }
    }
    ctx.redraw()
    return
  }
}

module.exports = {
  init,
  getLayoutTier,
  isClipped,
  getClippedBanner,
  getStatusText,
  getShortcutsText,
  getConversationNodes,
  getStreamNodes,
  getPlanNodes,
  onSubmitInput,
  onGlobalKey,
}
