const test = require('node:test')
const assert = require('node:assert')
const {
  headerBuffer,
  streamInputFrame,
  abortFrame,
  memQueryFrame,
  memCommitFrame,
  configFrame,
  snapshotSaveFrame,
  snapshotLoadFrame,
  resumeFrame,
  toolReturnFrame,
  taskTriageFrame,
  readStreamOpenFrame,
  readStreamCloseFrame,
  backlogTriageFrame,
  parseEventRouted,
  parseReadStreamStatus,
  parseBacklogRouted,
  pingFrame,
  parsedFrame,
} = require('../lib/protocol/framing')
const {
  MAGIC,
  OP_STREAM_INPUT,
  OP_ABORT,
  OP_MEM_QUERY,
  OP_MEM_COMMIT,
  OP_SET_CONFIG,
  OP_TOOL_RETURN,
  OP_SNAPSHOT_SAVE,
  OP_SNAPSHOT_LOAD,
  OP_RESUME,
  OP_TASK_TRIAGE,
  OP_READ_STREAM_OPEN,
  OP_READ_STREAM_CLOSE,
  OP_BACKLOG_TRIAGE,
  OP_BACKLOG_ROUTED,
  BACKLOG_ACTION_ACK,
  READ_STATUS_CHUNK,
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

test('memCommitFrame serializes commit opcode properly', () => {
  const frame = memCommitFrame(12)
  const parsed = parsedFrame(frame)
  assert.notStrictEqual(parsed, null)
  assert.strictEqual(parsed.header.msgId, 12)
  assert.strictEqual(parsed.header.opcode, OP_MEM_COMMIT)
  assert.strictEqual(parsed.payload.length, 0)
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

test('snapshotSaveFrame and snapshotLoadFrame serialize properly', () => {
  const saveBuf = snapshotSaveFrame('/tmp/snap.bin', 'str-123', 5)
  const parsedSave = parsedFrame(saveBuf)
  assert.strictEqual(parsedSave.header.opcode, OP_SNAPSHOT_SAVE)
  assert.strictEqual(parsedSave.header.msgId, 5)

  const loadBuf = snapshotLoadFrame('/tmp/snap.bin', 6)
  const parsedLoad = parsedFrame(loadBuf)
  assert.strictEqual(parsedLoad.header.opcode, OP_SNAPSHOT_LOAD)
  assert.strictEqual(parsedLoad.header.msgId, 6)
  assert.strictEqual(parsedLoad.payload.toString('utf-8'), '/tmp/snap.bin')
})

test('resumeFrame serializes properly with zero-length payload', () => {
  const frame = resumeFrame(8)
  const parsed = parsedFrame(frame)
  assert.strictEqual(parsed.header.opcode, OP_RESUME)
  assert.strictEqual(parsed.header.msgId, 8)
  assert.strictEqual(parsed.header.payloadLen, 0)
  assert.strictEqual(parsed.payload.length, 0)
})

test('toolReturnFrame serializes properly', () => {
  const frame = toolReturnFrame('ack', { status: 'ok' }, 7, 0, 12)
  const parsed = parsedFrame(frame)
  assert.strictEqual(parsed.header.opcode, OP_TOOL_RETURN)
  assert.strictEqual(parsed.header.msgId, 12)
  assert.strictEqual(parsed.payload.readUInt16LE(0), 7) // callId
  assert.strictEqual(parsed.payload.readUInt16LE(2), 0) // status
  assert.strictEqual(parsed.payload.readUInt16LE(4), 3) // name_len ('ack')
  assert.strictEqual(parsed.payload.subarray(6, 9).toString('utf-8'), 'ack')
  assert.strictEqual(parsed.payload.subarray(9).toString('utf-8'), JSON.stringify({ status: 'ok' }))
})

test('parsedFrame handles incomplete buffers gracefully', () => {
  const frame = pingFrame(1)
  const partial = frame.subarray(0, 10)
  assert.strictEqual(parsedFrame(partial), null)
})

test('taskTriageFrame serializes candidate tasks and event text', () => {
  const tasks = [
    { id: 1, title: 'Refactor parser' },
    { id: 2, title: 'Analyze memory dump' },
  ]
  const frame = taskTriageFrame(105, tasks, 'No Python please', 14)
  const parsed = parsedFrame(frame)
  assert.strictEqual(parsed.header.opcode, OP_TASK_TRIAGE)
  assert.strictEqual(parsed.header.msgId, 14)
  assert.strictEqual(parsed.payload.readUInt16LE(0), 105) // eventId
  assert.strictEqual(parsed.payload.readUInt16LE(2), 2) // numTasks
})

test('readStreamOpenFrame and readStreamCloseFrame serialize properly', () => {
  const openBuf = readStreamOpenFrame(3, 1024n, 'src/main.zig', 15)
  const parsedOpen = parsedFrame(openBuf)
  assert.strictEqual(parsedOpen.header.opcode, OP_READ_STREAM_OPEN)
  assert.strictEqual(parsedOpen.header.msgId, 15)
  assert.strictEqual(parsedOpen.payload.readUInt16LE(0), 3) // taskId
  assert.strictEqual(parsedOpen.payload.readBigUInt64LE(2), 1024n) // offset
  const pathLen = parsedOpen.payload.readUInt16LE(10)
  assert.strictEqual(parsedOpen.payload.subarray(12, 12 + pathLen).toString('utf-8'), 'src/main.zig')

  const closeBuf = readStreamCloseFrame(3, 16)
  const parsedClose = parsedFrame(closeBuf)
  assert.strictEqual(parsedClose.header.opcode, OP_READ_STREAM_CLOSE)
  assert.strictEqual(parsedClose.header.msgId, 16)
  assert.strictEqual(parsedClose.payload.readUInt16LE(0), 3)
})

test('parseEventRouted and parseReadStreamStatus unpack binary payloads', () => {
  const routedBuf = Buffer.alloc(6)
  routedBuf.writeUInt16LE(105, 0)
  routedBuf.writeUInt16LE(2, 2)
  routedBuf.writeUInt8(0, 4) // isNewTask = false
  const routed = parseEventRouted(routedBuf)
  assert.strictEqual(routed.eventId, 105)
  assert.strictEqual(routed.taskId, 2)
  assert.strictEqual(routed.isNewTask, false)

  const statusBuf = Buffer.alloc(16)
  statusBuf.writeUInt16LE(4, 0) // taskId
  statusBuf.writeUInt8(READ_STATUS_CHUNK, 2) // status
  statusBuf.writeUInt32LE(512, 4) // bytesRead
  statusBuf.writeBigUInt64LE(2048n, 8) // newOffset
  const status = parseReadStreamStatus(statusBuf)
  assert.strictEqual(status.taskId, 4)
  assert.strictEqual(status.status, READ_STATUS_CHUNK)
  assert.strictEqual(status.bytesRead, 512)
  assert.strictEqual(status.newOffset, 2048n)
})

test('backlogTriageFrame and parseBacklogRouted serialize and unpack properly', () => {
  const frame = backlogTriageFrame(102, 'Summarize large file', 17)
  const parsed = parsedFrame(frame)
  assert.strictEqual(parsed.header.opcode, OP_BACKLOG_TRIAGE)
  assert.strictEqual(parsed.header.msgId, 17)
  assert.strictEqual(parsed.payload.readUInt16LE(0), 102) // eventId
  const titleLen = parsed.payload.readUInt16LE(2)
  assert.strictEqual(parsed.payload.subarray(4, 4 + titleLen).toString('utf-8'), 'Summarize large file')

  const routedBuf = Buffer.alloc(4)
  routedBuf.writeUInt16LE(102, 0)
  routedBuf.writeUInt8(BACKLOG_ACTION_ACK, 2)
  const routed = parseBacklogRouted(routedBuf)
  assert.strictEqual(routed.eventId, 102)
  assert.strictEqual(routed.action, BACKLOG_ACTION_ACK)
})
