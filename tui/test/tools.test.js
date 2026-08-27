const test = require('node:test')
const assert = require('node:assert')
const { ToolRegistry } = require('../lib/tools')
const { ToolParser, parseToolCall } = require('../lib/tools/parser')

test('parseToolCall extracts tool name and json arguments', () => {
  const input = 'Let me check: <|tool_call>call:terminal_write{"input": "cargo build\n", "watch": "completion"}<tool_call|>'
  const parsed = parseToolCall(input)
  assert.notStrictEqual(parsed, null)
  assert.strictEqual(parsed.name, 'terminal_write')
  assert.strictEqual(parsed.args.input, 'cargo build\n')
  assert.strictEqual(parsed.args.watch, 'completion')
})

test('ToolRegistry executes terminal_write and key tools', async () => {
  let written = ''
  let keySent = ''
  const mockSession = {
    write: (txt) => { written += txt },
    sendKey: (k) => { keySent = k },
    buffer: { screenText: () => 'terminal screen ok' },
  }
  const mockClient = { sendInput: () => {} }
  const mockTimers = {}

  const registry = ToolRegistry(mockSession, mockClient, mockTimers)
  const res1 = await registry.execute('terminal_write', { input: 'git status\n' })
  assert.strictEqual(res1.status, 'written')
  assert.strictEqual(written, 'git status\n')

  const res2 = await registry.execute('terminal_key', { key: 'ctrl+c' })
  assert.strictEqual(res2.status, 'key_sent')
  assert.strictEqual(keySent, 'ctrl+c')

  let memQueried = ''
  let memTopK = 0
  const mockClient2 = {
    sendInput: () => {},
    sendMemQuery: (q, k) => {
      memQueried = q
      memTopK = k
    },
  }
  const registry2 = ToolRegistry(mockSession, mockClient2, mockTimers)
  const res3 = await registry2.execute('recall', { query: 'archived thoughts on memory', top_k: 3 })
  assert.strictEqual(res3.status, 'query_submitted')
  assert.strictEqual(memQueried, 'archived thoughts on memory')
  assert.strictEqual(memTopK, 3)
})

test('ToolParser intercepts streaming tool call and pushes response', async () => {
  let sentResponse = ''
  const mockRegistry = {
    execute: async (name, args) => ({ output: `executed ${name}` }),
  }
  const mockClient = {
    sendInput: (payload) => { sentResponse = payload },
  }

  const parser = ToolParser(mockRegistry, mockClient)
  await parser.ingestChunk('I will run this: <|tool_call>call:terminal_reset{}<tool_call|>')

  assert.strictEqual(sentResponse.includes('<|tool_response>'), true)
  assert.strictEqual(sentResponse.includes('executed terminal_reset'), true)
})
