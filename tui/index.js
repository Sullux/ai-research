const fs = require('fs')
const path = require('path')
const { Tui } = require('@sullux/tui')
const { Client } = require('./lib/client')
const { TerminalSession } = require('./lib/terminal/session')
const { TimerManager } = require('./lib/timers')
const { ToolRegistry } = require('./lib/tools')
const { ToolParser } = require('./lib/tools/parser')
const { StateStore } = require('./lib/ui/state')
const controller = require('./lib/ui/controller')

const loadConfig = () => {
  const cfgPath = path.resolve(__dirname, './config.json')
  if (fs.existsSync(cfgPath)) {
    try {
      return JSON.parse(fs.readFileSync(cfgPath, 'utf-8'))
    } catch (_) {}
  }
  return {
    modelPath: '../../gemma-4-12B-it',
    extraArgs: ['--gpu', '--q4', '--quiescence'],
    runtime: { maxTokens: 512, budget: 512, temp: 0.7, topP: 0.95, qThresh: 0.001 },
  }
}

const main = () => {
  const config = loadConfig()
  const cliArgs = process.argv.slice(2)
  const extraArgs = cliArgs.length > 0 ? cliArgs : config.extraArgs
  const modelPath = path.resolve(__dirname, config.modelPath)

  const store = StateStore()
  const session = TerminalSession({ rows: 30, cols: 100 })
  const client = Client({
    binaryPath: path.resolve(__dirname, '../zig-out/bin/infer'),
    modelPath,
    extraArgs,
  })
  const timers = TimerManager()
  const registry = ToolRegistry(session, client, timers)
  const parser = ToolParser(registry, client)

  controller.init(store, client, session)

  const app = Tui({
    view: path.resolve(__dirname, './view.yaml'),
    modules: { controller },
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
  if (config.runtime) client.setConfig(config.runtime)
  app.start()

  setInterval(() => {
    app.redraw()
  }, 250)
}

main()
