const { refs } = require('./state')
const {
  formatTurn1,
  formatUserTurn,
  formatUserDecisionTurn,
  formatTruncatedTurn,
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
      { isTurnContext: true, isTruncated: savedMsg.isTruncated },
    )
    refs.activeTurnNotificationId = notItem?.id || null
    eventId = notItem?.id || savedMsg.id
    if (savedMsg.isTruncated) {
      turnHeader = `[Event: ${eventId} | Source: ${savedMsg.relPath} | ${savedMsg.tokenCount} tok | read: ${savedMsg.relPath}]\n`
      turnBody = `${savedMsg.preview}... [Truncated. Use read({ path: "${savedMsg.relPath}" }) to inspect full content]`
    } else {
      turnHeader = `[Event: ${eventId} | Source: ${savedMsg.relPath}]\n`
      turnBody = savedMsg.payload
    }
  }

  // If currently generating an assistant response, stage this message as pendingInterjection
  // so the conversation view preserves causal sequence (Response 1 -> Interjection -> Response 2).
  const isGeneratingResponse = Boolean(refs.store?.state?.activeResponse)

  const turnContent = `${turnHeader}${turnBody}`
  refs.store?.pushHistory(val)

  if (isGeneratingResponse) {
    refs.store?.setPendingInterjection({ sender: 'User', text: val, time: Date.now() })
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
  } else if (savedMsg?.isTruncated) {
    payloadText = formatTruncatedTurn(
      `${alertsRollup}${turnContent}`,
      eventId,
      savedMsg.relPath,
    )
  } else {
    const waitingTasks = refs.orchestrator?.getWaitingForUserTasks?.() || []
    payloadText = waitingTasks.length > 0
      ? formatUserDecisionTurn(`${alertsRollup}${turnContent}`, waitingTasks)
      : formatUserTurn(`${alertsRollup}${turnContent}`)
  }

  refs.store?.setGenerating(true)
  refs.client.sendInput(payloadText)
  ctx.redraw()
}

module.exports = {
  onSubmitInput,
}
