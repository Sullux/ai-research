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

  refs.store?.pushHistory(val)
  refs.store?.addConversationMessage({ sender: 'User', text: val })
  refs.store?.addStreamEntry({ type: 'user', title: '👤 USER', content: val })

  refs.store?.setEditMode(false)
  ctx.setFocus?.(null)

  let payloadText = ''
  if (!refs.hasSentFirstTurn && refs.systemPrompt) {
    refs.hasSentFirstTurn = true
    payloadText = formatTurn1(refs.systemPrompt, val)
  } else {
    const waitingTasks = refs.orchestrator?.getWaitingForUserTasks?.() || []
    payloadText = waitingTasks.length > 0
      ? formatUserDecisionTurn(val, waitingTasks)
      : formatUserTurn(val)
  }

  refs.store?.setGenerating(true)
  refs.client.sendInput(payloadText)
  ctx.redraw()
}

module.exports = {
  onSubmitInput,
}
