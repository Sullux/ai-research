const { refs } = require('./state')

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
    payloadText = `<|turn>system\n<|think|>\n${refs.systemPrompt}\n<turn|>\n<|turn>user\n${val}\n<turn|>\n<|turn>model\n`
  } else {
    payloadText = `<|turn>user\n${val}\n<turn|>\n<|turn>model\n`
  }

  refs.store?.setGenerating(true)
  refs.client.sendInput(payloadText)
  ctx.redraw()
}

module.exports = {
  onSubmitInput,
}
