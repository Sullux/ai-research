const { refs } = require('./state')

const getLayoutTier = (cols) => {
  const width = cols || refs.store?.state?.dimensions?.cols || 100
  if (width >= 200) return 1
  if (width >= 160) return 2
  return 3
}

const isClipped = (cols, rows) => {
  const c = cols || refs.store?.state?.dimensions?.cols || 100
  const r = rows || refs.store?.state?.dimensions?.rows || 30
  return c < 80 || r < 30
}

const getClippedBanner = () =>
  ' ⚠️  TERMINAL TOO SMALL (Minimum 80x30 required) — CONTENT CLIPPED '

const getStatusText = () => {
  if (!refs.store) return 'Status: Ready'
  const state = refs.store.state
  const pausedTag = state.isPaused ? ' [⏸️ PAUSED]' : ''
  return ` ${state.status}${pausedTag}`
}

const getShortcutsText = () => {
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
    return ' [j/k] Scroll | [x] Expand/Collapse | [a] Chat | [d] Plan | [c] Copy | [Esc] Normal/Pause'
  }
  if (state.mode === 'plan') {
    return ' [j/k] Navigate Tasks | [a] Chat | [s] Stream | [c] Copy | [Esc] Normal/Pause'
  }
  return ' [Enter] Type | [a] Chat | [s] Stream | [d] Plan | [Ctrl+Q] Quit'
}

module.exports = {
  getLayoutTier,
  isClipped,
  getClippedBanner,
  getStatusText,
  getShortcutsText,
}
