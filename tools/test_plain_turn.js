const net = require('net')
const path = require('path')
const { spawn } = require('child_process')
const { clientFactory } = require('../tui/lib/client')
const { EventEmitter } = require('events')

async function run() {
  const binaryPath = path.resolve(__dirname, '../zig-out/bin/infer')
  const modelPath = path.resolve(__dirname, '../../gemma-4-12B-it-qat-q4_0-unquantized')
  const client = clientFactory(spawn, EventEmitter)({
    binaryPath,
    modelPath,
    extraArgs: ['--gpu', '--q4', '--quiescence'],
  })

  let totalDecoded = 0
  let startTime = 0
  let isThought = false

  client.on('thought', ({ text }) => {
    if (!startTime) startTime = Date.now()
    process.stdout.write(`\x1b[33m${text}\x1b[0m`)
    totalDecoded++
  })

  client.on('content', ({ text }) => {
    if (!startTime) startTime = Date.now()
    process.stdout.write(`\x1b[32m${text}\x1b[0m`)
    totalDecoded++
  })

  client.on('status', (s) => {
    if (s.status === 2) { // GENERATING
      // console.log(`[STATUS GENERATING] tokSec: ${s.tokSec.toFixed(1)}, currentTok: ${s.currentTok}`)
    }
  })

  client.on('turnComplete', (t) => {
    console.log(`\n\n[TURN COMPLETE] totalTok: ${t.totalTok}, elapsedMs: ${t.elapsedMs} ms, tokSec: ${t.tokSec.toFixed(2)} tok/s`)
    client.shutdown()
    process.exit(0)
  })

  client.start()
  await new Promise(r => setTimeout(r, 1000))

  client.setConfig({ budget: 512, temp: 0.0, topP: 0.95, qThresh: 0.0, maxTokens: 256 })
  await new Promise(r => setTimeout(r, 100))

  console.log('[SENDING PLAIN USER TEXT: "How are you doing today?"]')
  client.sendInput("How are you doing today?")
}

run()
