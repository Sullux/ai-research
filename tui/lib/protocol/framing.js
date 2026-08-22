const {
  MAGIC,
  PROTOCOL_VERSION,
  OP_STREAM_INPUT,
  OP_ABORT,
  OP_MEM_QUERY,
  OP_SET_CONFIG,
  OP_PING,
  OP_SHUTDOWN,
  MODE_TEXT,
} = require('./constants')

const headerBuffer = (opcode, msgId, payloadLen) => {
  const buf = Buffer.alloc(16)
  buf.writeUInt32LE(MAGIC, 0)
  buf.writeUInt16LE(PROTOCOL_VERSION, 4)
  buf.writeUInt16LE(msgId, 6)
  buf.writeUInt16LE(opcode, 8)
  buf.writeUInt16LE(0, 10)
  buf.writeUInt32LE(payloadLen, 12)
  return buf
}

const streamInputFrame = (text, msgId = 1) => {
  const textBytes = Buffer.from(text, 'utf-8')
  const payload = Buffer.alloc(8 + textBytes.length)
  payload.writeUInt8(MODE_TEXT, 0)
  textBytes.copy(payload, 8)
  const hdr = headerBuffer(OP_STREAM_INPUT, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const abortFrame = (msgId = 0) => headerBuffer(OP_ABORT, msgId, 0)

const pingFrame = (msgId = 1) => headerBuffer(OP_PING, msgId, 0)

const shutdownFrame = () => headerBuffer(OP_SHUTDOWN, 0, 0)

const memQueryFrame = (query, msgId = 1, topK = 5) => {
  const queryBytes = Buffer.from(query, 'utf-8')
  const payload = Buffer.alloc(24 + queryBytes.length)
  payload.writeUInt8(0x00, 0) // keywords
  payload.writeUInt8(topK, 1)
  payload.writeBigUInt64LE(0n, 8)
  payload.writeBigUInt64LE(0n, 16)
  queryBytes.copy(payload, 24)
  const hdr = headerBuffer(OP_MEM_QUERY, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const configFrame = (budget = 512, temp = 1.0, topP = 0.95, qThresh = 0.001, maxTok = 64, msgId = 1) => {
  const payload = Buffer.alloc(20)
  payload.writeUInt32LE(budget, 0)
  payload.writeFloatLE(temp, 4)
  payload.writeFloatLE(topP, 8)
  payload.writeFloatLE(qThresh, 12)
  payload.writeUInt32LE(maxTok, 16)
  const hdr = headerBuffer(OP_SET_CONFIG, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const parsedFrame = (buf) => {
  if (buf.length < 16) return null
  const magic = buf.readUInt32LE(0)
  if (magic !== MAGIC) return null
  const version = buf.readUInt16LE(4)
  const msgId = buf.readUInt16LE(6)
  const opcode = buf.readUInt16LE(8)
  const payloadLen = buf.readUInt32LE(12)
  if (buf.length < 16 + payloadLen) return null

  const payload = buf.subarray(16, 16 + payloadLen)
  const rest = buf.subarray(16 + payloadLen)
  return { header: { version, msgId, opcode, payloadLen }, payload, rest }
}

module.exports = {
  headerBuffer,
  streamInputFrame,
  abortFrame,
  pingFrame,
  shutdownFrame,
  memQueryFrame,
  configFrame,
  parsedFrame,
}
