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

const formattedTurn = (text) => {
  if (isFirstTurn) {
    isFirstTurn = false
    const kernelPath = path.resolve(__dirname, '../../PROMPT_KERNEL.md')
    const kernel = fs.existsSync(kernelPath) ? fs.readFileSync(kernelPath, 'utf-8').trim() : ''
    return kernel.length > 0
      ? `<|turn>system\n<|think|>\n${kernel}\n<turn|>\n<|turn>user\n${text}<turn|>\n<|turn>model\n`
      : `<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n${text}<turn|>\n<|turn>model\n`
  }
  return `<|turn>user\n${text}<turn|>\n<|turn>model\n`
}

const submitPrompt = () => {
  const text = store?.state.input?.trim()
  if (!text || !client) return
  store.appendDialogue(`\n> ${text}\n`)
  store.setInput('')
  store.setGenerating(true)
  client.sendInput(formattedTurn(text))
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
