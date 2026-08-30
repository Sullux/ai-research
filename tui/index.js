const fs = require('fs')
const path = require('path')
const { Tui } = require('@sullux/tui')
const { Client } = require('./lib/client')
const { TerminalSession } = require('./lib/terminal/session')
const { TimerManager } = require('./lib/timers')
const { Orchestrator } = require('./lib/orchestrator')
const { ToolRegistry } = require('./lib/tools')
const { ToolParser } = require('./lib/tools/parser')
const { StateStore } = require('./lib/ui/state')
const controller = require('./lib/ui/controller')

const STATUS_NAMES = [
  'Idle',
  'Encoding prompt...',
  'Generating...',
  'Searching memory...',
  'Consolidating diffs...',
]

const loadConfig = () => {
  const cfgPath = path.resolve(__dirname, './config.json')
  if (fs.existsSync(cfgPath)) {
    try {
      return JSON.parse(fs.readFileSync(cfgPath, 'utf-8'))
    } catch (_) {}
  }
  return {
    modelPath: '../../gemma-4-12B-it',
    promptPath: './PROMPT.md',
    extraArgs: ['--gpu', '--q4', '--quiescence'],
    runtime: {
      maxTokens: 512,
      thinkingBudget: 512,
      temp: 0.7,
      topP: 0.95,
      qThresh: 0.001,
    },
  }
}

const loadSystemPrompt = (promptPath) => {
  const kernelPath = path.resolve(__dirname, './PROMPT_KERNEL.md')
  const userPromptPath = path.resolve(__dirname, promptPath || './PROMPT.md')

  const kernel = fs.existsSync(kernelPath) ? fs.readFileSync(kernelPath, 'utf-8').trim() : ''
  const userPrompt = fs.existsSync(userPromptPath) ? fs.readFileSync(userPromptPath, 'utf-8').trim() : ''

  return [userPrompt, kernel].filter(Boolean).join('\n\n')
}

const main = () => {
  const config = loadConfig()
  const cliArgs = process.argv.slice(2)
  const extraArgs = cliArgs.length > 0 ? cliArgs : config.extraArgs
  const modelPath = path.resolve(__dirname, config.modelPath)
  const systemPrompt = loadSystemPrompt(config.promptPath)

  const store = StateStore()
  const session = TerminalSession({ rows: 30, cols: 100 })
  const client = Client({
    binaryPath: path.resolve(__dirname, '../zig-out/bin/infer'),
    modelPath,
    extraArgs,
  })
  const timers = TimerManager()
  const orchestrator = Orchestrator(timers)
  const registry = ToolRegistry(session, client, timers, orchestrator)
  const parser = ToolParser(registry, client)

  controller.init(store, client, session, orchestrator, timers, systemPrompt)

  const app = Tui({
    view: path.resolve(__dirname, './view.yaml'),
    modules: { controller },
    autoFocus: false,
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
    const clean = text.replaceAll('\u2581', ' ')
    store.appendActiveThought(clean)
    requestRedraw()
  })

  client.on('content', ({ text }) => {
    const clean = text.replaceAll('\u2581', ' ')
    store.appendActiveResponse(clean)
    parser.ingestChunk(clean)
    requestRedraw()
  })

  client.on('status', ({ status, isGpu, activeSlots, archivedDiffs, tokSec, currentTok, totalTok }) => {
    const sName = STATUS_NAMES[status] || 'Active'
    const isMixed = extraArgs.includes('--mixed')
    const devTag = isGpu ? (isMixed ? 'GPU Mixed' : 'GPU Q4_0') : 'CPU BF16'
    const prog = status === 1
      ? ` (${currentTok}/${totalTok} tok)`
      : (status === 2 ? ` (${currentTok} tok)` : '')
    const rate = tokSec > 0 ? ` | ${tokSec.toFixed(1)} tok/s` : ''
    store.setStatus(
      `[${devTag}] ${sName}${prog}${rate} | Slots: ${activeSlots} | Memory: ${archivedDiffs} diffs`,
    )
    requestRedraw()
  })

  client.on('turnComplete', ({ tokSec, elapsedMs, totalTok }) => {
    store.flushActiveThought()
    store.flushActiveResponse()
    store.setGenerating(false)
    store.setStatus(`Idle | ${tokSec.toFixed(1)} tok/s | ${totalTok} tok in ${elapsedMs}ms`)

    // Autonomous tick loop check
    const activeStep = orchestrator.getActiveStep()
    if (activeStep && activeStep.status === 'IN_PROGRESS') {
      const nextThought = orchestrator.buildThoughtPrefix('STEP_TICK', { step: activeStep })
      store.setGenerating(true)
      client.sendInput(nextThought)
    }

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

  setInterval(() => {
    app.redraw()
  }, 250)
}

main()
