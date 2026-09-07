const { refs } = require('./state')

const getLayoutTier = (cols) => {
  const width = cols || refs.store?.state?.dimensions?.cols || (typeof process !== 'undefined' && process.stdout?.columns) || 100
  if (width >= 160) return 1
  if (width >= 120) return 2
  return 3
}

const isClipped = (cols, rows) => {
  const c = cols || refs.store?.state?.dimensions?.cols || (typeof process !== 'undefined' && process.stdout?.columns) || 100
  const r = rows || refs.store?.state?.dimensions?.rows || (typeof process !== 'undefined' && process.stdout?.rows) || 30
  return c < 60 || r < 30
}

const getClippedBanner = () =>
  ' ⚠️  TERMINAL TOO SMALL (Minimum 60x30 required) — CONTENT CLIPPED '

const isConversationVisible = () => {
  const tier = getLayoutTier()
  if (tier <= 2) return true
  const mode = refs.store?.state?.mode || 'chat'
  return mode === 'chat'
}

const getConversationWidth = () => {
  const tier = getLayoutTier()
  if (tier === 1) return '40%'
  if (tier === 2) {
    const mode = refs.store?.state?.mode || 'chat'
    return mode === 'plan' ? '60%' : '50%'
  }
  const mode = refs.store?.state?.mode || 'chat'
  return mode === 'chat' ? '100%' : '0%'
}

const isStreamVisible = () => {
  const tier = getLayoutTier()
  if (tier === 1) return true
  const mode = refs.store?.state?.mode || 'chat'
  if (tier === 2) return mode !== 'plan'
  return mode === 'stream'
}

const getStreamWidth = () => {
  const tier = getLayoutTier()
  if (tier === 1) return '40%'
  if (tier === 2) {
    const mode = refs.store?.state?.mode || 'chat'
    return mode === 'plan' ? '0%' : '50%'
  }
  const mode = refs.store?.state?.mode || 'chat'
  return mode === 'stream' ? '100%' : '0%'
}

const getStreamMargin = () => {
  if (!isStreamVisible()) return 0
  const convVisible = isConversationVisible()
  return convVisible ? { left: 1 } : 0
}

const isPlanVisible = () => {
  const tier = getLayoutTier()
  if (tier === 1) return true
  const mode = refs.store?.state?.mode || 'chat'
  return mode === 'plan'
}

const getPlanWidth = () => {
  const tier = getLayoutTier()
  if (tier === 1) return '20%'
  if (tier === 2) {
    const mode = refs.store?.state?.mode || 'chat'
    return mode === 'plan' ? '40%' : '0%'
  }
  const mode = refs.store?.state?.mode || 'chat'
  return mode === 'plan' ? '100%' : '0%'
}

const getPlanMargin = () => {
  if (!isPlanVisible()) return 0
  const otherVisible = isConversationVisible() || isStreamVisible()
  return otherVisible ? { left: 1 } : 0
}

const getStatusText = () => {
  if (!refs.store) return 'Status: Ready'
  const state = refs.store.state
  const pausedTag = state.isPaused ? ' [⏸️ PAUSED]' : ''
  const pendingCount = refs.notManager?.getUnserviced?.()?.length || 0
  const alertsTag = pendingCount > 0 ? ` | [🔔 ${pendingCount} ALERTS]` : ''
  return ` ${state.status}${pausedTag}${alertsTag}`
}

const getShortcutsText = () => {
  if (isClipped()) {
    return getClippedBanner()
  }
  if (!refs.store) return ' [Enter] Type  [a] Chat  [s] Stream  [d] Plan  [Ctrl+Q] Quit'
  const state = refs.store.state
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
    return ' [j/k] Scroll | [x] Expand/Collapse | [a] Chat | [d] Plan | [c] Copy | [Esc] Chat'
  }
  if (state.mode === 'plan') {
    return ' [j/k] Navigate Tasks | [a] Chat | [s] Stream | [c] Copy | [Esc] Chat'
  }
  return ' [Enter] Type | [a] Chat | [s] Stream | [d] Plan | [Ctrl+Q] Quit'
}

module.exports = {
  getLayoutTier,
  isClipped,
  getClippedBanner,
  isConversationVisible,
  getConversationWidth,
  isStreamVisible,
  getStreamWidth,
  getStreamMargin,
  isPlanVisible,
  getPlanWidth,
  getPlanMargin,
  getStatusText,
  getShortcutsText,
}
