const fs = require('fs')
const path = require('path')

let store = null
let client = null
let session = null
let isFirstTurn = true

const init = (s, c, sess) => {
  store = s
  client = c
  session = sess
  isFirstTurn = true
}

const getThoughts = () => store?.state.thoughts || '(Thinking scratchpad)'
const getDialogue = () => store?.state.dialogue || ''
const getTerminal = () => session?.buffer.screenText() || ''
const getStatus = () => store?.state.status || 'Status: Ready'
const getInput = () => `> ${store?.state.input || ''}█`

const submitPrompt = () => {
  const text = store?.state.input?.trim()
  if (!text || !client) return
  store.clearThoughts?.()
  store.appendDialogue(`\n> ${text}\n`)
  store.setInput('')
  store.setGenerating(true)
  client.sendInput(text)
}

const abortGeneration = () => {
  if (client) client.sendAbort()
  store?.setGenerating(false)
  store?.appendDialogue('\n[Interrupted via OP_ABORT]\n')
}

const onKey = (ctx, event) => {
  if (event.ctrl && event.key === 'x') {
    abortGeneration()
    return
  }
  if (event.key === 'enter') {
    submitPrompt()
    return
  }
  if (event.key === 'backspace') {
    const cur = store?.state.input || ''
    if (cur.length > 0) store.setInput(cur.slice(0, -1))
    return
  }
  if (event.char && event.char.length === 1 && !event.ctrl && !event.alt) {
    const cur = store?.state.input || ''
    store.setInput(cur + event.char)
  }
}

module.exports = {
  init,
  getThoughts,
  getDialogue,
  getTerminal,
  getStatus,
  getInput,
  submitPrompt,
  abortGeneration,
  onKey,
}
