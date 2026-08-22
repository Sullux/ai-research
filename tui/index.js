const path = require('path')
const { Tui } = require('@sullux/tui')
const { Client } = require('./lib/client')
const { TerminalSession } = require('./lib/terminal/session')
const { TimerManager } = require('./lib/timers')
const { ToolRegistry } = require('./lib/tools')
const { ToolParser } = require('./lib/tools/parser')
const { StateStore } = require('./lib/ui/state')
const controller = require('./lib/ui/controller')

const main = () => {
  const store = StateStore()
  const session = TerminalSession({ rows: 30, cols: 100 })
  const client = Client({
    binaryPath: path.resolve(__dirname, '../zig-out/bin/infer'),
    modelPath: path.resolve(__dirname, '../../gemma-4-E2B'),
    extraArgs: process.argv.slice(2),
  })
  const timers = TimerManager()
  const registry = ToolRegistry(session, client, timers)
  const parser = ToolParser(registry, client)

  controller.init(store, client, session)

  const app = Tui({
    view: path.resolve(__dirname, './view.yaml'),
    truecolor: true,
  })

  client.on('thought', ({ text }) => {
    store.appendThought(text)
    app.redraw()
  })

  client.on('content', ({ text }) => {
    store.appendDialogue(text)
    parser.ingestChunk(text)
    app.redraw()
  })

  client.on('turnComplete', ({ tokSec, elapsedMs, totalTok }) => {
    store.setGenerating(false)
    store.setStatus(`Idle | ${tokSec.toFixed(1)} tok/s | ${totalTok} tok in ${elapsedMs}ms`)
    app.redraw()
  })

  client.on('memResponse', ({ count, status }) => {
    store.setStatus(`Memory retrieved: ${count} episodes (status: ${status})`)
    app.redraw()
  })

  process.stdin.on('keypress', (_, key) => {
    if (!key) return
    if (key.ctrl && key.name === 'c') {
      session.kill()
      client.shutdown()
      process.exit(0)
    }
    if (key.ctrl && key.name === 'x') {
      controller.abortGeneration()
      return
    }
    if (key.name === 'return') {
      controller.submitPrompt()
    } else if (key.name === 'backspace') {
      const cur = store.state.input
      if (cur.length > 0) store.setInput(cur.slice(0, -1))
    } else if (key.sequence && key.sequence.length === 1 && !key.ctrl && !key.meta) {
      store.setInput(store.state.input + key.sequence)
    }
    app.redraw()
  })

  client.start()
  app.start()

  setInterval(() => {
    app.redraw()
  }, 250)
}

main()
