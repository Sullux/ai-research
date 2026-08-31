const test = require('node:test')
const assert = require('node:assert')
const {
  headerBuffer,
  streamInputFrame,
  abortFrame,
  memQueryFrame,
  configFrame,
  pingFrame,
  parsedFrame,
} = require('../lib/protocol/framing')
const {
  MAGIC,
  OP_STREAM_INPUT,
  OP_ABORT,
  OP_MEM_QUERY,
  OP_SET_CONFIG,
  OP_PING,
} = require('../lib/protocol/constants')

test('headerBuffer creates 16-byte valid binary header', () => {
  const hdr = headerBuffer(OP_STREAM_INPUT, 42, 100)
  assert.strictEqual(hdr.length, 16)
  assert.strictEqual(hdr.readUInt32LE(0), MAGIC)
  assert.strictEqual(hdr.readUInt16LE(4), 1) // version
  assert.strictEqual(hdr.readUInt16LE(6), 42) // msgId
  assert.strictEqual(hdr.readUInt16LE(8), OP_STREAM_INPUT) // opcode
  assert.strictEqual(hdr.readUInt32LE(12), 100) // payloadLen
})

test('streamInputFrame serializes text payload properly', () => {
  const frame = streamInputFrame('hello world', 7)
  const parsed = parsedFrame(frame)
  assert.notStrictEqual(parsed, null)
  assert.strictEqual(parsed.header.msgId, 7)
  assert.strictEqual(parsed.header.opcode, OP_STREAM_INPUT)
  assert.strictEqual(parsed.payload.readUInt8(0), 0) // MODE_TEXT
  assert.strictEqual(parsed.payload.subarray(8).toString('utf-8'), 'hello world')
})

test('memQueryFrame serializes query and topK correctly', () => {
  const frame = memQueryFrame('matrix multiplication', 9, 3)
  const parsed = parsedFrame(frame)
  assert.notStrictEqual(parsed, null)
  assert.strictEqual(parsed.header.msgId, 9)
  assert.strictEqual(parsed.header.opcode, OP_MEM_QUERY)
  assert.strictEqual(parsed.payload.readUInt8(1), 3) // topK
  assert.strictEqual(parsed.payload.subarray(24).toString('utf-8'), 'matrix multiplication')
})

test('configFrame serializes penalty and runtime options properly', () => {
  const frame = configFrame(512, 0.7, 0.95, 0.001, 256, 0.05, 1.15, 64, 0.2, 0.3, 11)
  const parsed = parsedFrame(frame)
  assert.notStrictEqual(parsed, null)
  assert.strictEqual(parsed.header.msgId, 11)
  assert.strictEqual(parsed.header.opcode, OP_SET_CONFIG)
  assert.strictEqual(parsed.payload.length, 40)
  assert.strictEqual(parsed.payload.readUInt32LE(0), 512)
  assert.strictEqual(parsed.payload.readUInt32LE(16), 256)
  assert.strictEqual(parsed.payload.readUInt32LE(28), 64)
})

test('parsedFrame handles incomplete buffers gracefully', () => {
  const frame = pingFrame(1)
  const partial = frame.subarray(0, 10)
  assert.strictEqual(parsedFrame(partial), null)
})
