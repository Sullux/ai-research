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

  const snooze = (id, duration = null) => {
    let item = pending.get(id)
    if (!item && snoozed.has(id)) {
      item = snoozed.get(id)
      if (item.timerId) timers?.cancelTimer?.(item.timerId)
    }
    if (!item) return { status: 'not_found', id }

    pending.delete(id)

    // Duration-free snooze: queue deferral to the bottom of the pending LIFO stack
    if (!duration || duration === 'idle' || duration === 'defer') {
      const deferredItem = {
        ...item,
        status: 'PENDING',
        isDeferred: true,
      }
      // Put at the very bottom (first key in map iteration)
      const currentEntries = Array.from(pending.entries())
      pending.clear()
      pending.set(id, deferredItem)
      for (const [k, v] of currentEntries) {
        pending.set(k, v)
      }
      return { status: 'deferred_to_tail', id }
    }

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
    item.isDeferred = false
    pending.set(id, item)
  }

  const markServicing = (id) => {
    const item = pending.get(id)
    if (item) {
      item.status = 'SERVICING'
      return item
    }
    return null
  }

  // Returns in LIFO order (most recent interrupt first)
  const getPending = () => Array.from(pending.values()).reverse()

  // Returns only unserviced interrupts in LIFO order
  const getUnserviced = () => getPending().filter((item) => item.status === 'PENDING' && !item.isDeferred)

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
    markServicing,
    getPending,
    getUnserviced,
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
