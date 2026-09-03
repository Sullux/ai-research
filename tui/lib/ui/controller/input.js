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

  // Flush any in-flight thought or response before recording barge-in user turn
  if (refs.store?.state?.activeThought) {
    refs.store.flushActiveThought()
  }
  if (refs.store?.state?.activeResponse) {
    refs.store.flushActiveResponse()
  }

  refs.store?.pushHistory(val)
  refs.store?.addConversationMessage({ sender: 'User', text: val })
  refs.store?.addStreamEntry({ type: 'user', title: '👤 USER', content: val })

  refs.store?.setEditMode(false)
  ctx.setFocus?.(null)

  // VFS message persistence & notification generation
  let turnContent = val
  if (refs.vfs) {
    const savedMsg = refs.vfs.saveUserMessage(val)
    if (savedMsg.isTruncated) {
      // Create notification alert for large inputs
      refs.notManager?.notify(
        savedMsg.relPath,
        savedMsg.preview,
        savedMsg.id,
      )
      turnContent = `[🔔 ${savedMsg.id} (${savedMsg.relPath} | ${savedMsg.tokenCount} tok): "${savedMsg.preview}..." | read: ${savedMsg.relPath}]`
    } else {
      // Short message: deliver 100% content directly in notification banner
      turnContent = `[🔔 ${savedMsg.relPath}: "${savedMsg.payload}"]`
    }
  }

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
  refs.client.sendInput(payloadText)
  ctx.redraw()
}

module.exports = {
  onSubmitInput,
}
