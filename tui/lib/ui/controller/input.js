const { refs } = require('./state')
const {
  formatTurn1,
  formatUserTurn,
  formatUserDecisionTurn,
} = require('../../template')
const { Afsm, chatMachineFactory } = require('../../afsm')

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
  const waiting = refs.orchestrator?.getWaitingForUserTasks?.() || []
  const payloadText = waiting.length > 0
    ? formatUserDecisionTurn(`${alertsRollup}${turnContent}`, waiting)
    : formatUserTurn(`${alertsRollup}${turnContent}`)

  refs.store?.setGenerating(true)
  if (!refs.isEngineReady) {
    refs.pendingInputTurn = payloadText
    ctx.redraw()
    return
  }

  let dispatchPromise = null
  if (!isGenerating || !refs.notManager) {
    if (typeof refs.client?.probe === 'function') {
      const afsm = Afsm({
        machine: chatMachineFactory(),
        client: refs.client,
        initialContext: { val },
      })

      dispatchPromise = afsm.step()
        .then((stepResult) => {
          const isDirect = stepResult?.op === 'direct_response'
          const meta = stepResult?.transition?.meta || {}
          const confPct = meta.confidence != null ? (meta.confidence * 100).toFixed(1) : '?'
          if (isDirect) {
            refs.store?.addStreamEntry({
              type: 'notice',
              title: '⚡ THINKING GATE',
              content: `Immediate response chosen (${confPct}% confidence >= 80%). Bypassing reasoning.`,
            })
          } else {
            const reason = meta.winningIdx === 1 ? 'essential reasoning' : `confidence ${confPct}% < 80% threshold`
            refs.store?.addStreamEntry({
              type: 'notice',
              title: '🧠 THINKING GATE',
              content: `Deliberate reasoning engaged (${reason}).`,
            })
          }
          refs.client.sendInput(payloadText, { direct: isDirect })
          ctx.redraw?.()
        })
        .catch((_) => {
          refs.client.sendInput(payloadText, { direct: false })
          ctx.redraw?.()
        })
    } else {
      refs.client.sendInput(payloadText, { direct: false })
    }
  }
  ctx.redraw?.()
  return dispatchPromise
}

module.exports = {
  onSubmitInput,
}
