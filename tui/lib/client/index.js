const { spawn } = require('child_process')
const { EventEmitter } = require('events')
const {
  OP_STREAM_CONTENT, OP_STREAM_THOUGHT, OP_TURN_COMPLETE,
  OP_MEM_RESPONSE, OP_STATUS, OP_PONG, OP_ERROR,
} = require('../protocol/constants')
const {
  streamInputFrame, abortFrame, memQueryFrame, configFrame, shutdownFrame, parsedFrame,
} = require('../protocol/framing')

const clientFactory = (spawnProc, EmitterClass) => (opts) => {
  const { binaryPath, modelPath, extraArgs = [] } = opts
  const emitter = new EmitterClass()
  let proc = null
  let rxBuf = Buffer.alloc(0)
  let nextMsgId = 1

  const handleFrame = (h, p) => {
    const text = () => p.subarray(24).toString('utf-8').replace(/\u2581/g, ' ')
    if (h.opcode === OP_STREAM_CONTENT) emitter.emit('content', { text: text(), msgId: h.msgId })
    else if (h.opcode === OP_STREAM_THOUGHT) emitter.emit('thought', { text: text(), msgId: h.msgId })
    else if (h.opcode === OP_TURN_COMPLETE) emitter.emit('turnComplete', {
      totalTok: p.readUInt32LE(0), elapsedMs: p.readUInt32LE(4), tokSec: p.readFloatLE(8), reason: p.readUInt8(12), msgId: h.msgId,
    })
    else if (h.opcode === OP_STATUS) emitter.emit('status', {
      status: p.readUInt8(0), isGpu: p.readUInt8(1), activeSlots: p.readUInt16LE(2), archivedDiffs: p.readUInt16LE(4),
      tokSec: p.readFloatLE(8), currentTok: p.readUInt32LE(12), totalTok: p.readUInt32LE(16), msgId: h.msgId,
    })
    else if (h.opcode === OP_MEM_RESPONSE) emitter.emit('memResponse', { count: p.readUInt16LE(0), status: p.readUInt8(2), msgId: h.msgId })
    else if (h.opcode === OP_PONG) emitter.emit('pong', { msgId: h.msgId })
    else if (h.opcode === OP_ERROR) emitter.emit('error', { error: p.toString('utf-8'), msgId: h.msgId })
  }

  const start = () => {
    proc = spawnProc(binaryPath, ['--model', modelPath, '--serve', ...extraArgs], { stdio: ['pipe', 'pipe', 'inherit'] })
    proc.stdout.on('data', (chunk) => {
      rxBuf = Buffer.concat([rxBuf, chunk])
      let parsed = parsedFrame(rxBuf)
      let count = 0
      while (parsed) {
        rxBuf = parsed.rest
        handleFrame(parsed.header, parsed.payload)
        count++
        parsed = parsedFrame(rxBuf)
      }
      if (count > 0) emitter.emit('drain')
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

  const setConfig = (o) => {
    if (proc?.stdin?.writable) {
      proc.stdin.write(configFrame(o.budget, o.temp, o.topP, o.qThresh, o.maxTokens, nextMsgId++))
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
