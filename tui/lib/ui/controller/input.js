const { refs } = require('./state')
const {
  formatTurn1,
  formatUserTurn,
  formatUserDecisionTurn,
} = require('../../template')

const onSubmitInput = (ctx, payload) => {
  const val = payload.value?.trim()
  if (payload.node) {
    payload.node.value = ''
    payload.node.cursor = 0
  }
  if (!val || !refs.client) return

  // Flush in-flight thought before recording barge-in user turn.
  // We do NOT flush activeResponse here if it's currently generating tokens,
  // allowing the ongoing micro-burst to complete and place the interjection in causal order.
  if (refs.store?.state?.activeThought) {
    refs.store.flushActiveThought()
  }

  // If currently generating an assistant response, stage this message as pendingInterjection
  // so the conversation view preserves causal sequence (Response 1 -> Interjection -> Response 2).
  const isGeneratingResponse = Boolean(refs.store?.state?.activeResponse)
  const isGenerating = Boolean(
    refs.store?.state?.isGenerating ||
    refs.store?.state?.activeResponse ||
    refs.store?.state?.activeThought
  )

  // VFS message persistence & notification generation
  let turnHeader = ''
  let turnBody = val
  let savedMsg = null
  let eventId = null
  if (refs.vfs) {
    savedMsg = refs.vfs.saveUserMessage(val)

    const notItem = refs.notManager?.notify(
      savedMsg.relPath,
      savedMsg.preview,
      savedMsg.id,
      { isTurnContext: true, payload: savedMsg.payload },
    )
    if (isGenerating && refs.activeTurnNotificationId) {
      refs.interruptedTurnNotificationId = refs.activeTurnNotificationId
    }
    if (!isGenerating && notItem) {
      refs.notManager?.markServicing(notItem.id)
      refs.activeTurnNotificationId = notItem.id
    }
    eventId = notItem?.id || savedMsg.id
    turnHeader = `[Event: ${eventId} | Source: ${savedMsg.relPath}]\n`
    turnBody = savedMsg.payload

    // Autonomic task triage: if existing active tasks exist, query 1-token probe
    const activeTasks = refs.taskManager?.getActiveTasks() || []
    if (activeTasks.length > 0 && refs.client) {
      const triageSeq = notItem ? notItem.seq : parseInt(savedMsg.id, 10)
      refs.client.sendTaskTriage(triageSeq, refs.taskManager.formatCandidates(), val)
    } else if (refs.taskManager) {
      const firstLine = val.trim().split('\n')[0]
      const fallbackTitle = firstLine.length > 80 ? firstLine.slice(0, 77) + '...' : firstLine
      const newTask = refs.taskManager.createTask(fallbackTitle)
      if (refs.client) {
        refs.client.sendTaskTitle(newTask.id, val)
      }
    }

    if (!refs.taskManager && refs.activeTurnNotificationId && isGenerating) {
      refs.notManager?.suspend(refs.activeTurnNotificationId)
      refs.activeTurnNotificationId = null
    }
  }

  const turnContent = `${turnHeader}${turnBody}`
  refs.store?.pushHistory(val)

  if (isGeneratingResponse) {
    refs.store?.setPendingInterjection({ sender: 'User', text: val, time: Date.now() })
  } else if (!refs.isEngineReady) {
    refs.store?.addConversationMessage({ sender: 'User', text: val, waitingEngine: true })
  } else {
    refs.store?.addConversationMessage({ sender: 'User', text: val })
  }
  refs.store?.addStreamEntry({ type: 'user', title: '👤 USER', content: turnContent })

  refs.store?.setEditMode(false)
  ctx.setFocus?.(null)

  // Prepend any pending alerts rollup
  const alertsRollup = refs.notManager?.formatTurnAlerts?.() || ''

  let payloadText = ''
  if (!refs.hasSentFirstTurn && refs.systemPrompt) {
    refs.hasSentFirstTurn = true
    payloadText = formatTurn1(refs.systemPrompt, `${alertsRollup}${turnContent}`)
  } else {
    const waitingTasks = refs.orchestrator?.getWaitingForUserTasks?.() || []
    payloadText = waitingTasks.length > 0
      ? formatUserDecisionTurn(`${alertsRollup}${turnContent}`, waitingTasks)
      : formatUserTurn(`${alertsRollup}${turnContent}`)
  }

  refs.store?.setGenerating(true)
  if (!refs.isEngineReady) {
    // Engine still loading weights or restoring snapshot: stage turn to dispatch immediately upon ready
    refs.pendingInputTurn = payloadText
  } else if (!isGenerating || !refs.notManager) {
    refs.client.sendInput(payloadText)
  }
  ctx.redraw()
}

module.exports = {
  onSubmitInput,
}
