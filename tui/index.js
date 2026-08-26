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

const STATUS_NAMES = ['Idle', 'Encoding prompt...', 'Generating...', 'Searching memory...', 'Consolidating diffs...']

const loadConfig = () => {
  const cfgPath = path.resolve(__dirname, './config.json')
  if (fs.existsSync(cfgPath)) {
    try { return JSON.parse(fs.readFileSync(cfgPath, 'utf-8')) } catch (_) {}
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

  let isDirty = false
  let redrawPending = false
  const requestRedraw = () => {
    isDirty = true
    if (redrawPending) return
    redrawPending = true
    setImmediate(() => {
      redrawPending = false
      if (isDirty) {
        isDirty = false
        app.redraw()
      }
    })
  }

  client.on('thought', ({ text }) => {
    store.appendThought(text.replaceAll('\u2581', ' '))
  })

  client.on('content', ({ text }) => {
    const clean = text.replaceAll('\u2581', ' ')
    store.appendDialogue(clean)
    parser.ingestChunk(clean)
  })

  client.on('status', ({ status, isGpu, activeSlots, archivedDiffs, tokSec, currentTok, totalTok }) => {
    const sName = STATUS_NAMES[status] || 'Active'
    const isMixed = extraArgs.includes('--mixed')
    const devTag = isGpu ? (isMixed ? 'GPU Mixed' : 'GPU Q4_0') : 'CPU BF16'
    const prog = status === 1
      ? ` (${currentTok}/${totalTok} tok)`
      : (status === 2 ? ` (${currentTok} tok)` : '')
    const rate = tokSec > 0 ? ` | ${tokSec.toFixed(1)} tok/s` : ''
    store.setStatus(`[${devTag}] ${sName}${prog}${rate} | Slots: ${activeSlots} | Memory: ${archivedDiffs} diffs`)
  })

  client.on('turnComplete', ({ tokSec, elapsedMs, totalTok }) => {
    store.setGenerating(false)
    store.setStatus(`Idle | ${tokSec.toFixed(1)} tok/s | ${totalTok} tok in ${elapsedMs}ms`)
    requestRedraw()
  })

  client.on('memResponse', ({ count, status }) => {
    store.setStatus(`Memory retrieved: ${count} episodes (status: ${status})`)
    requestRedraw()
  })

  client.on('drain', requestRedraw)

  app.onExit(() => {
    session.kill()
    client.shutdown()
  })

  client.start()
  if (config.runtime) client.setConfig(config.runtime)
  app.start()

  setInterval(() => { app.redraw() }, 250)
}

main()
