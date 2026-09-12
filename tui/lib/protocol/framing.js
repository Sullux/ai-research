const {
  MAGIC,
  PROTOCOL_VERSION,
  OP_STREAM_INPUT,
  OP_ABORT,
  OP_MEM_QUERY,
  OP_SET_CONFIG,
  OP_TOOL_RETURN,
  OP_MEM_COMMIT,
  OP_SET_SYSTEM,
  OP_SNAPSHOT_SAVE,
  OP_SNAPSHOT_LOAD,
  OP_RESUME,
  OP_TASK_TRIAGE,
  OP_READ_STREAM_OPEN,
  OP_READ_STREAM_CLOSE,
  OP_BACKLOG_TRIAGE,
  OP_TASK_TITLE,
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

const memCommitFrame = (msgId = 1) => headerBuffer(OP_MEM_COMMIT, msgId, 0)

const pingFrame = (msgId = 1) => headerBuffer(OP_PING, msgId, 0)

const shutdownFrame = () => headerBuffer(OP_SHUTDOWN, 0, 0)

const setSystemFrame = (systemJson, msgId = 1) => {
  const jsonBytes = Buffer.from(typeof systemJson === 'string' ? systemJson : JSON.stringify(systemJson), 'utf-8')
  const hdr = headerBuffer(OP_SET_SYSTEM, msgId, jsonBytes.length)
  return Buffer.concat([hdr, jsonBytes])
}

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

const configFrame = (
  thinkingBudget = 512,
  temp = 1.0,
  topP = 0.95,
  qThresh = 0.001,
  maxTok = 64,
  minP = 0.05,
  repeatPenalty = 1.1,
  repeatLastN = 64,
  frequencyPenalty = 0.1,
  presencePenalty = 0.1,
  msgId = 1,
) => {
  const payload = Buffer.alloc(40)
  payload.writeUInt32LE(thinkingBudget, 0)
  payload.writeFloatLE(temp, 4)
  payload.writeFloatLE(topP, 8)
  payload.writeFloatLE(qThresh, 12)
  payload.writeUInt32LE(maxTok, 16)
  payload.writeFloatLE(minP, 20)
  payload.writeFloatLE(repeatPenalty, 24)
  payload.writeUInt32LE(repeatLastN, 28)
  payload.writeFloatLE(frequencyPenalty, 32)
  payload.writeFloatLE(presencePenalty, 36)
  const hdr = headerBuffer(OP_SET_CONFIG, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const snapshotSaveFrame = (snapPath, streamId = '', msgId = 1) => {
  const pathBytes = Buffer.from(snapPath, 'utf-8')
  const idBytes = Buffer.from(streamId, 'utf-8')
  const payload = Buffer.alloc(2 + pathBytes.length + 2 + idBytes.length)
  payload.writeUInt16LE(pathBytes.length, 0)
  pathBytes.copy(payload, 2)
  payload.writeUInt16LE(idBytes.length, 2 + pathBytes.length)
  idBytes.copy(payload, 2 + pathBytes.length + 2)
  const hdr = headerBuffer(OP_SNAPSHOT_SAVE, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const snapshotLoadFrame = (snapPath, msgId = 1) => {
  const pathBytes = Buffer.from(snapPath, 'utf-8')
  const hdr = headerBuffer(OP_SNAPSHOT_LOAD, msgId, pathBytes.length)
  return Buffer.concat([hdr, pathBytes])
}

const resumeFrame = (msgId = 1) => headerBuffer(OP_RESUME, msgId, 0)

const taskTriageFrame = (eventId, tasks = [], eventText = '', msgId = 1) => {
  const eventBytes = Buffer.from(eventText, 'utf-8')
  let taskBytesTotal = 0
  const encodedTasks = tasks.map((t) => {
    const titleBytes = Buffer.from(t.title ?? '', 'utf-8')
    taskBytesTotal += 4 + titleBytes.length
    return { id: t.id, titleBytes }
  })
  const payload = Buffer.alloc(4 + taskBytesTotal + 2 + eventBytes.length)
  payload.writeUInt16LE(eventId, 0)
  payload.writeUInt16LE(encodedTasks.length, 2)
  let off = 4
  for (const t of encodedTasks) {
    payload.writeUInt16LE(t.id, off)
    payload.writeUInt16LE(t.titleBytes.length, off + 2)
    t.titleBytes.copy(payload, off + 4)
    off += 4 + t.titleBytes.length
  }
  payload.writeUInt16LE(eventBytes.length, off)
  eventBytes.copy(payload, off + 2)
  const hdr = headerBuffer(OP_TASK_TRIAGE, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const readStreamOpenFrame = (taskId, offset = 0n, path = '', msgId = 1) => {
  const pathBytes = Buffer.from(path, 'utf-8')
  const payload = Buffer.alloc(2 + 8 + 2 + pathBytes.length)
  payload.writeUInt16LE(taskId, 0)
  payload.writeBigUInt64LE(BigInt(offset), 2)
  payload.writeUInt16LE(pathBytes.length, 10)
  pathBytes.copy(payload, 12)
  const hdr = headerBuffer(OP_READ_STREAM_OPEN, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const readStreamCloseFrame = (taskId, msgId = 1) => {
  const payload = Buffer.alloc(2)
  payload.writeUInt16LE(taskId, 0)
  const hdr = headerBuffer(OP_READ_STREAM_CLOSE, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const backlogTriageFrame = (eventId, title = '', msgId = 1) => {
  const titleBytes = Buffer.from(title || '', 'utf-8')
  const payload = Buffer.alloc(4 + titleBytes.length)
  payload.writeUInt16LE(eventId, 0)
  payload.writeUInt16LE(titleBytes.length, 2)
  titleBytes.copy(payload, 4)
  const hdr = headerBuffer(OP_BACKLOG_TRIAGE, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const taskTitleFrame = (taskId, text = '', msgId = 1) => {
  const textBytes = Buffer.from(text || '', 'utf-8')
  const payload = Buffer.alloc(4 + 2 + textBytes.length)
  payload.writeUInt32LE(taskId, 0)
  payload.writeUInt16LE(textBytes.length, 4)
  textBytes.copy(payload, 6)
  const hdr = headerBuffer(OP_TASK_TITLE, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const parseTaskTitleResult = (payload) => {
  if (payload.length < 6) return null
  const taskId = payload.readUInt32LE(0)
  const titleLen = payload.readUInt16LE(4)
  if (payload.length < 6 + titleLen) return null
  const title = payload.slice(6, 6 + titleLen).toString('utf-8')
  return { taskId, title }
}

const parseBacklogRouted = (payload) => {
  const eventId = payload.readUInt16LE(0)
  const action = payload.readUInt8(2)
  return { eventId, action }
}

const parseEventRouted = (payload) => {
  const eventId = payload.readUInt16LE(0)
  const taskId = payload.readUInt16LE(2)
  const isNewTask = Boolean(payload.readUInt8(4))
  return { eventId, taskId, isNewTask }
}

const parseReadStreamStatus = (payload) => {
  const taskId = payload.readUInt16LE(0)
  const status = payload.readUInt8(2)
  const bytesRead = payload.readUInt32LE(4)
  const newOffset = payload.readBigUInt64LE(8)
  return { taskId, status, bytesRead, newOffset }
}

const toolReturnFrame = (toolName, result, callId = 1, status = 0, msgId = 1) => {
  const nameBytes = Buffer.from(toolName, 'utf-8')
  const jsonStr = typeof result === 'string' ? result : JSON.stringify(result)
  const jsonBytes = Buffer.from(jsonStr, 'utf-8')
  const payload = Buffer.alloc(6 + nameBytes.length + jsonBytes.length)
  payload.writeUInt16LE(callId, 0)
  payload.writeUInt16LE(status, 2)
  payload.writeUInt16LE(nameBytes.length, 4)
  nameBytes.copy(payload, 6)
  jsonBytes.copy(payload, 6 + nameBytes.length)
  const hdr = headerBuffer(OP_TOOL_RETURN, msgId, payload.length)
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
  memCommitFrame,
  setSystemFrame,
  pingFrame,
  shutdownFrame,
  resumeFrame,
  memQueryFrame,
  toolReturnFrame,
  snapshotSaveFrame,
  snapshotLoadFrame,
  taskTriageFrame,
  readStreamOpenFrame,
  readStreamCloseFrame,
  backlogTriageFrame,
  taskTitleFrame,
  parseEventRouted,
  parseReadStreamStatus,
  parseBacklogRouted,
  parseTaskTitleResult,
  configFrame,
  parsedFrame,
}
