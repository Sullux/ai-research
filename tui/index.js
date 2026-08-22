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
    modules: {
      controller,
    },
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

  app.onExit(() => {
    session.kill()
    client.shutdown()
  })

  client.start()
  app.start()

  setInterval(() => {
    app.redraw()
  }, 250)
}

main()
