const formatTimestamp = (ts) => {
  const d = new Date(ts)
  const pad = (n) => String(n).padStart(2, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

const refs = {
  store: null,
  client: null,
  session: null,
  orchestrator: null,
  timers: null,
  systemPrompt: '',
  hasSentFirstTurn: false,
  activeTurnNotificationId: null,
  vfs: null,
  notManager: null,
}

const init = (s, c, sess, orch, tmrs, sysPrompt = '', vfs = null, notManager = null) => {
  refs.store = s
  refs.client = c
  refs.session = sess
  refs.orchestrator = orch
  refs.timers = tmrs
  refs.systemPrompt = sysPrompt
  refs.hasSentFirstTurn = false
  refs.vfs = vfs
  refs.notManager = notManager
}

module.exports = {
  refs,
  init,
  formatTimestamp,
}
