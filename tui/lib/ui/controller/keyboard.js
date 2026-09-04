const fs = require('fs')
const path = require('path')
const { KeyHandler } = require('@sullux/tui')
const { refs } = require('./state')
const { getLayoutTier } = require('./layout')
const { copyToClipboard, getSelectedText } = require('./clipboard')

const logDebug = (msg) => {
  try {
    fs.appendFileSync(path.resolve(process.cwd(), 'debug.log'), `[${new Date().toISOString()}] ${msg}\n`)
  } catch (_) {}
}

const toggleOverlay = (store, key, tier, minTier) => {
  store?.setMode(key)
  if (tier >= minTier) store?.setOverlay(store.state.overlay === key ? null : key)
}

const scrollList = (store, delta) => {
  const mode = store?.state.mode
  const selKey = mode === 'chat' ? 'chat' : 'stream'
  let totalCount = 0
  if (mode === 'chat') {
    totalCount = (store?.state.conversation?.length || 0) + (store?.state.activeResponse ? 1 : 0)
  } else if (mode === 'stream') {
    totalCount = (store?.state.stream?.length || 0) + (store?.state.activeThought ? 1 : 0) + (store?.state.activeResponse ? 1 : 0)
  } else {
    totalCount = store?.state.plan?.length || 0
  }
  const next = (store?.state.selectedIdx[selKey] || 0) + delta
  if (next >= 0 && next < totalCount) {
    store.state.selectedIdx[selKey] = next
    if (delta < 0) store.state.stickyScroll[selKey] = false
    else if (next === totalCount - 1) store.state.stickyScroll[selKey] = true
  }
}

const normalModeRouter = KeyHandler({
  'ctrl+q': () => process.exit(0),
  'ctrl+c': () => process.exit(0),
  a: (ctx, e) => { e.stopPropagation(); refs.store?.setMode('chat'); refs.store?.setOverlay(null); ctx.redraw() },
  s: (ctx, e) => { e.stopPropagation(); toggleOverlay(refs.store, 'stream', getLayoutTier(ctx.width), 3); ctx.redraw() },
  d: (ctx, e) => { e.stopPropagation(); toggleOverlay(refs.store, 'plan', getLayoutTier(ctx.width), 2); ctx.redraw() },
  enter: (ctx, e) => {
    e.stopPropagation()
    refs.store?.setEditMode(true)
    refs.store?.setMode('chat')
    ctx.setFocus?.('chatInput')
    ctx.redraw()
  },
  escape: (ctx, e) => {
    e.stopPropagation()
    if (refs.store?.state.overlay) {
      refs.store.setOverlay(null)
    } else {
      const p = !refs.store?.state.isPaused
      refs.store?.setPaused(p)
      if (p) refs.client?.sendAbort?.()
    }
    ctx.redraw()
  },
  r: (ctx, e) => {
    if (refs.store?.state.isPaused) {
      e.stopPropagation()
      refs.store.setPaused(false)
      ctx.redraw()
    }
  },
  x: (ctx, e) => {
    if (refs.store?.state.mode === 'stream') {
      e.stopPropagation()
      refs.store.toggleExpandStreamItem(refs.store.state.selectedIdx.stream)
      ctx.redraw()
    }
  },
  c: (ctx, e) => {
    e.stopPropagation()
    const text = getSelectedText(refs.store?.state)
    if (text) copyToClipboard(text)
  },
  j: (ctx, e) => { e.stopPropagation(); scrollList(refs.store, 1); ctx.redraw() },
  k: (ctx, e) => { e.stopPropagation(); scrollList(refs.store, -1); ctx.redraw() },
  down: (ctx, e) => normalModeRouter(ctx, { ...e, key: 'j' }),
  up: (ctx, e) => normalModeRouter(ctx, { ...e, key: 'k' }),
})

const onGlobalKey = (ctx, event) => {
  logDebug(`onGlobalKey: stroke="${event.stroke}" key="${event.key}" ctrl=${event.ctrl} isEditMode=${refs.store?.state.isEditMode}`)
  if ((event.ctrl && (event.key === 'q' || event.key === 'c')) || event.stroke === 'ctrl+q') {
    logDebug('onGlobalKey matched exit sequence. Exiting...')
    process.exit(0)
  }

  if (refs.store?.state.isEditMode) {
    if (event.key === 'escape') {
      event.stopPropagation()
      refs.store.setEditMode(false)
      ctx.setFocus?.(null)
      ctx.redraw()
      return
    }
    return
  }

  normalModeRouter(ctx, event)
  event.stopPropagation()
}

module.exports = { onGlobalKey }
