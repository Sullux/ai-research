let store = null
let client = null
let session = null

const init = (s, c, sess) => {
  store = s
  client = c
  session = sess
}

const getThoughts = () => store?.state.thoughts || '(Thinking scratchpad)'
const getDialogue = () => store?.state.dialogue || ''
const getTerminal = () => session?.buffer.screenText() || ''
const getStatus = () => store?.state.status || 'Status: Ready'
const getInput = () => store?.state.input || ''

const submitPrompt = () => {
  const text = store?.state.input?.trim()
  if (!text || !client) return
  store.appendDialogue(`\n> ${text}\n`)
  store.setInput('')
  store.setGenerating(true)
  client.sendInput(text)
}

const abortGeneration = () => {
  if (client) client.sendAbort()
  store?.setGenerating(false)
  store?.appendDialogue('\n[Interrupted]\n')
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
}
