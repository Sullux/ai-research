const parseDurationMs = (str) => {
  if (!str || typeof str !== 'string') return 60000
  const match = str.trim().match(/^(\d+(?:\.\d+)?)\s*(ms|s|m|h)?$/i)
  if (!match) return 60000
  const val = parseFloat(match[1])
  const unit = (match[2] || 's').toLowerCase()
  if (unit === 'ms') return Math.round(val)
  if (unit === 's') return Math.round(val * 1000)
  if (unit === 'm') return Math.round(val * 60 * 1000)
  if (unit === 'h') return Math.round(val * 3600 * 1000)
  return 60000
}

const notificationManagerFactory = (now) => (timers) => {
  let notCounter = 100
  const pending = new Map()
  const snoozed = new Map()

  const notify = (source, preview, refId = null, extra = {}) => {
    notCounter += 1
    const id = `not_${notCounter}`
    const item = {
      id,
      source, // e.g. '/msg/user/msg_1042.txt' or '/sys/cmd/cmd_101'
      preview: typeof preview === 'string' ? preview : JSON.stringify(preview || ''),
      refId, // e.g. 'msg_1042', 'cmd_101'
      extra,
      timestamp: now(),
      status: 'PENDING',
    }
    pending.set(id, item)
    return item
  }

  const ack = (id) => {
    if (pending.has(id)) {
      const item = pending.get(id)
      pending.delete(id)
      return { status: 'acknowledged', id, item }
    }
    if (snoozed.has(id)) {
      const item = snoozed.get(id)
      if (item.timerId) timers?.cancelTimer?.(item.timerId)
      snoozed.delete(id)
      return { status: 'acknowledged', id, item }
    }
    return { status: 'not_found', id }
  }

  const snooze = (id, duration = '1m') => {
    let item = pending.get(id)
    if (!item && snoozed.has(id)) {
      item = snoozed.get(id)
      if (item.timerId) timers?.cancelTimer?.(item.timerId)
    }
    if (!item) return { status: 'not_found', id }

    pending.delete(id)
    const ms = parseDurationMs(duration)
    const sec = Math.max(1, Math.round(ms / 1000))

    let timerId = null
    if (typeof timers?.setTimer === 'function') {
      const res = timers.setTimer(sec, `snooze_${id}`, () => {
        wake(id)
      })
      timerId = res.timerId
    }

    const snoozedItem = {
      ...item,
      status: 'SNOOZED',
      duration,
      snoozedAt: now(),
      wakeAt: now() + ms,
      timerId,
    }
    snoozed.set(id, snoozedItem)
    return { status: 'snoozed', id, duration }
  }

  const wake = (id) => {
    const item = snoozed.get(id)
    if (!item) return
    snoozed.delete(id)
    item.status = 'PENDING'
    pending.set(id, item)
  }

  const getPending = () => Array.from(pending.values())

  const getSnoozed = () => Array.from(snoozed.values())

  const formatTurnAlerts = () => {
    const list = getPending()
    if (list.length === 0) return ''
    const lines = list.map((item) => `  - ${item.id} (${item.source}): "${item.preview}"`)
    return `[🔔 PENDING ALERTS:\n${lines.join('\n')}\n]\n`
  }

  return {
    notify,
    ack,
    snooze,
    wake,
    getPending,
    getSnoozed,
    formatTurnAlerts,
  }
}

const NotificationManager = notificationManagerFactory(Date.now)

module.exports = {
  notificationManagerFactory,
  NotificationManager,
  parseDurationMs,
}
