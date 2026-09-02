const fs = require('fs')
const path = require('path')
const { Tui } = require('@sullux/tui')
const { Client } = require('./lib/client')
const { TerminalSession } = require('./lib/terminal/session')
const { TimerManager } = require('./lib/timers')
const { Orchestrator } = require('./lib/orchestrator')
const { ToolRegistry } = require('./lib/tools')
const { ToolParser } = require('./lib/tools/parser')
const { StreamLog } = require('./lib/storage')
const { stateStoreFactory } = require('./lib/ui/state')
const controller = require('./lib/ui/controller')

const STATUS_NAMES = [
  'Idle',
  'Encoding prompt...',
  'Generating...',
  'Searching memory...',
  'Consolidating diffs...',
]

const parseCliArgs = (argv) => {
  let configPath = null
  const extraArgs = []
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--config' || arg === '-c') {
      configPath = argv[++i]
    } else if (arg.startsWith('--config=')) {
      configPath = arg.slice('--config='.length)
    } else {
      extraArgs.push(arg)
    }
  }
  return { configPath, extraArgs }
}

const loadConfig = (explicitPath) => {
  const cfgPath = explicitPath
    ? path.resolve(process.cwd(), explicitPath)
    : path.resolve(__dirname, './config.json')
  const baseDir = path.dirname(cfgPath)

  let raw = undefined
  if (fs.existsSync(cfgPath)) {
    try {
      raw = JSON.parse(fs.readFileSync(cfgPath, 'utf-8'))
    } catch (_) {}
  }
  const base = Object.assign({
    modelPath: '../../gemma-4-12B-it',
    memoryDir: './.memory',
    promptPath: './PROMPT.md',
    extraArgs: ['--gpu', '--q4'],
    runtime: {
      thinkingBudget: 256,
      maxTokens: 1024,
      temp: 0.7,
      topP: 0.95,
      minP: 0.05,
      repeatPenalty: 1.1,
      repeatLastN: 64,
      frequencyPenalty: 0.1,
      presencePenalty: 0.1,
      qThresh: 0.001,
    }
  }, raw)

  return {
    ...base,
    modelPath: base.modelPath ? path.resolve(baseDir, base.modelPath) : undefined,
    memoryDir: base.memoryDir ? path.resolve(baseDir, base.memoryDir) : undefined,
    promptPath: base.promptPath ? path.resolve(baseDir, base.promptPath) : undefined,
  }
}

const loadSystemPrompt = (promptPath) => {
  const kernelPath = path.resolve(__dirname, './PROMPT_KERNEL.md')
  const userPromptPath = promptPath || path.resolve(__dirname, './PROMPT.md')

  const kernel = fs.existsSync(kernelPath)
    ? fs.readFileSync(kernelPath, 'utf-8').trim()
    : ''
  const userPrompt = fs.existsSync(userPromptPath)
    ? fs.readFileSync(userPromptPath, 'utf-8').trim()
    : ''

  return [userPrompt, kernel].filter(Boolean).join('\n\n')
}

const main = () => {
  const { configPath, extraArgs: cliExtraArgs } = parseCliArgs(process.argv.slice(2))
  const config = loadConfig(configPath)
  const extraArgs = [...(cliExtraArgs.length > 0 ? cliExtraArgs : config.extraArgs || [])]
  const modelPath = config.modelPath
  const systemPrompt = loadSystemPrompt(config.promptPath)

  let streamLog = StreamLog(undefined)
  if (config.memoryDir) {
    const memDir = config.memoryDir
    fs.mkdirSync(memDir, { recursive: true })
    const episodicMemPath = path.join(memDir, '.episodic.mem')
    if (!extraArgs.includes('--memory') && !extraArgs.includes('--storage')) {
      extraArgs.push('--memory', episodicMemPath)
    }
    streamLog = StreamLog(memDir)
  }

  const StateStore = stateStoreFactory()
  const store = StateStore(streamLog.append)
  const tail = streamLog.loadTail()
  if (tail.items?.length > 0) {
    store.hydrateFromStream(tail.items)
  }

  const session = TerminalSession({ rows: 30, cols: 100 })
  const client = Client({
    binaryPath: path.resolve(__dirname, '../zig-out/bin/infer'),
    modelPath,
    extraArgs,
  })
  const timers = TimerManager()
  const orchestrator = Orchestrator(timers)
  const registry = ToolRegistry(session, client, timers, orchestrator)
  const parser = ToolParser(registry, client, store)

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
    if (store.state.activeResponse) {
      store.flushActiveResponse()
    }
    store.appendActiveThought(clean)
    requestRedraw()
  })

  client.on('content', ({ text }) => {
    const clean = text.replaceAll('\u2581', ' ')
    if (store.state.activeThought) {
      store.flushActiveThought()
    }
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

  client.on('exit', ({ code }) => {
    store.setGenerating(false)
    store.setStatus(`[Error] Inference backend stopped (exit code: ${code})`)
    requestRedraw()
  })

  client.on('error', ({ error }) => {
    store.setGenerating(false)
    store.setStatus(`[Error] Backend process error: ${error}`)
    requestRedraw()
  })

  const cleanup = () => {
    try { streamLog.close() } catch (_) {}
    try { session.kill() } catch (_) {}
    try { client.shutdown() } catch (_) {}
    try { client.kill() } catch (_) {}
  }

  app.onExit(cleanup)
  process.on('exit', cleanup)
  process.on('SIGINT', () => { cleanup(); process.exit(0) })
  process.on('SIGTERM', () => { cleanup(); process.exit(0) })
  process.on('SIGHUP', () => { cleanup(); process.exit(0) })
  process.on('uncaughtException', (err) => {
    cleanup()
    console.error(err)
    process.exit(1)
  })

  client.start()
  if (config.runtime) client.setConfig(config.runtime)
  app.start()

  setInterval(() => {
    app.redraw()
  }, 250)
}

if (require.main === module) {
  main()
}

module.exports = {
  loadConfig,
  loadSystemPrompt,
  parseCliArgs,
  main,
}
