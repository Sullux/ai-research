const { spawn } = require('child_process')
const { EventEmitter } = require('events')
const {
  OP_STREAM_CONTENT,
  OP_STREAM_THOUGHT,
  OP_TURN_COMPLETE,
  OP_MEM_RESPONSE,
  OP_PONG,
  OP_ERROR,
} = require('../protocol/constants')
const {
  streamInputFrame,
  abortFrame,
  memQueryFrame,
  configFrame,
  shutdownFrame,
  parsedFrame,
} = require('../protocol/framing')

const clientFactory = (spawnProc = spawn, Emitter = EventEmitter) => (options = {}) => {
  const binaryPath = options.binaryPath || '../zig-out/bin/infer'
  const modelPath = options.modelPath || '../gemma-4-E2B'
  const extraArgs = options.extraArgs || []
  const emitter = new Emitter()
  let proc = null
  let rxBuf = Buffer.alloc(0)
  let nextMsgId = 1

  const handleFrame = (h, p) => {
    if (h.opcode === OP_STREAM_CONTENT) {
      emitter.emit('content', { text: p.subarray(24).toString('utf-8'), msgId: h.msgId })
    } else if (h.opcode === OP_STREAM_THOUGHT) {
      emitter.emit('thought', { text: p.subarray(24).toString('utf-8'), msgId: h.msgId })
    } else if (h.opcode === OP_TURN_COMPLETE) {
      emitter.emit('turnComplete', {
        totalTok: p.readUInt32LE(0), elapsedMs: p.readUInt32LE(4),
        tokSec: p.readFloatLE(8), reason: p.readUInt8(12), msgId: h.msgId,
      })
    } else if (h.opcode === OP_MEM_RESPONSE) {
      emitter.emit('memResponse', { count: p.readUInt16LE(0), status: p.readUInt8(2), msgId: h.msgId })
    } else if (h.opcode === OP_PONG) {
      emitter.emit('pong', { msgId: h.msgId })
    } else if (h.opcode === OP_ERROR) {
      emitter.emit('error', { error: p.toString('utf-8'), msgId: h.msgId })
    }
  }

  const start = () => {
    proc = spawnProc(binaryPath, ['--model', modelPath, '--serve', ...extraArgs], {
      stdio: ['pipe', 'pipe', 'inherit'],
    })
    proc.stdout.on('data', (chunk) => {
      rxBuf = Buffer.concat([rxBuf, chunk])
      let parsed = parsedFrame(rxBuf)
      while (parsed) {
        rxBuf = parsed.rest
        handleFrame(parsed.header, parsed.payload)
        parsed = parsedFrame(rxBuf)
      }
    })
    proc.on('close', (code) => emitter.emit('exit', { code }))
    proc.on('error', (err) => emitter.emit('error', { error: err.message }))
  }

  const sendInput = (text) => {
    const id = nextMsgId++
    if (proc?.stdin?.writable) proc.stdin.write(streamInputFrame(text, id))
    return id
  }

  const sendAbort = () => {
    if (proc?.stdin?.writable) proc.stdin.write(abortFrame(nextMsgId++))
  }

  const sendMemQuery = (query, topK = 5) => {
    const id = nextMsgId++
    if (proc?.stdin?.writable) proc.stdin.write(memQueryFrame(query, id, topK))
    return id
  }

  const setConfig = (opts) => {
    if (proc?.stdin?.writable) {
      proc.stdin.write(configFrame(opts.budget, opts.temp, opts.topP, opts.qThresh, opts.maxTokens, nextMsgId++))
    }
  }

  const shutdown = () => {
    if (proc?.stdin?.writable) proc.stdin.write(shutdownFrame())
  }

  return { start, sendInput, sendAbort, sendMemQuery, setConfig, shutdown, on: emitter.on.bind(emitter) }
}

module.exports = {
  clientFactory,
  Client: clientFactory(spawn, EventEmitter),
}
