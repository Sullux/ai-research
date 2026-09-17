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
  if (refs.store?.state?.activeThought) {
    refs.store.flushActiveThought()
  }

  // If currently generating an assistant response, stage this message as pendingInterjection
  // so the conversation view preserves causal sequence (Response 1 -> Interjection -> Response 2).
  const isGeneratingResponse = Boolean(refs.store?.state?.activeResponse)
  const isGenerating = Boolean(
    refs.store?.state?.isGenerating ||
    refs.store?.state?.activeResponse ||
    refs.store?.state?.activeThought,
  )

  let turnHeader = '', turnBody = val, savedMsg = null, eventId = null
  if (refs.vfs) {
    savedMsg = refs.vfs.saveUserMessage(val)
    refs.channelManager?.registerChannel({
      id: 'chat/user',
      path: savedMsg.relPath,
      type: 'push',
      cursor: 0,
      isFocused: true,
    })
    const chunk = refs.channelManager?.readChunk('chat/user', refs.vfs, 512)
    const readContent = chunk?.content || savedMsg.payload

    const notItem = refs.notManager?.notify(
      savedMsg.relPath,
      savedMsg.preview,
      savedMsg.id,
      { isTurnContext: true, payload: savedMsg.payload },
    )
    if (isGenerating && refs.activeTurnNotificationId) {
      refs.notManager?.suspend(refs.activeTurnNotificationId)
      refs.activeTurnNotificationId = null
    } else if (!isGenerating && notItem) {
      refs.notManager?.markServicing(notItem.id)
      refs.activeTurnNotificationId = notItem.id
    }
    eventId = notItem?.id || savedMsg.id
    turnHeader = `[Event: ${eventId} | Source: ${savedMsg.relPath}]\n`
    turnBody = readContent
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
  refs.store?.addStreamEntry({
    type: 'user',
    title: '👤 USER',
    content: turnContent,
    rawText: val,
  })
  refs.store?.setEditMode(false)
  ctx.setFocus?.(null)

  const alertsRollup = refs.notManager?.formatTurnAlerts?.() || ''
  let payloadText = ''
  if (!refs.hasSentFirstTurn && refs.systemPrompt) {
    refs.hasSentFirstTurn = true
    payloadText = formatTurn1(refs.systemPrompt, `${alertsRollup}${turnContent}`)
  } else {
    const waiting = refs.orchestrator?.getWaitingForUserTasks?.() || []
    payloadText = waiting.length > 0
      ? formatUserDecisionTurn(`${alertsRollup}${turnContent}`, waiting)
      : formatUserTurn(`${alertsRollup}${turnContent}`)
  }

  refs.store?.setGenerating(true)
  if (!refs.isEngineReady) refs.pendingInputTurn = payloadText
  else if (!isGenerating || !refs.notManager) refs.client.sendInput(payloadText)
  ctx.redraw()
}

module.exports = {
  onSubmitInput,
}
