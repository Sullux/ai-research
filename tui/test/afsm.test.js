const test = require('node:test')
const assert = require('node:assert')
const { Afsm } = require('../lib/afsm')

test('Afsm evaluates discrete probe and transitions state', async () => {
  const definition = {
    initial: 'idle',
    states: {
      idle: {
        probe: {
          prompt: 'Choose next action: 1 (read), 2 (wait)',
          candidates: [
            { key: '1', op: 'start_reading', target: 'reading' },
            { key: '2', op: 'wait', target: 'idle' },
          ],
        },
      },
      reading: {
        probe: {
          prompt: 'Reading: 1 (continue), 0 (done)',
          candidates: [
            { key: '1', op: 'continue_reading', target: 'reading' },
            { key: '0', op: 'finish', target: 'idle' },
          ],
        },
      },
    },
  }

  const mockClient = {
    probe: async (prompt, candidates) => {
      if (prompt.includes('Choose next action')) {
        return { winningIdx: 0, confidence: 0.95, entropy: 0.05, text: '' } // selects '1'
      }
      return { winningIdx: 1, confidence: 0.9, entropy: 0.1, text: '' } // selects '0'
    },
  }

  let started = false
  let finished = false
  const ops = {
    start_reading: (ctx) => {
      started = true
      return { offset: 0 }
    },
    finish: (ctx) => {
      finished = true
      return { offset: 1024 }
    },
  }

  const fsm = Afsm({ definition, client: mockClient, initialContext: {}, ops })
  assert.strictEqual(fsm.getState().currentState, 'idle')

  const res1 = await fsm.step()
  assert.strictEqual(res1.from, 'idle')
  assert.strictEqual(res1.to, 'reading')
  assert.strictEqual(res1.op, 'start_reading')
  assert.strictEqual(started, true)
  assert.strictEqual(fsm.getState().context.offset, 0)

  const res2 = await fsm.step()
  assert.strictEqual(res2.from, 'reading')
  assert.strictEqual(res2.to, 'idle')
  assert.strictEqual(res2.op, 'finish')
  assert.strictEqual(finished, true)
  assert.strictEqual(fsm.getState().context.offset, 1024)
})

test('Afsm supports generative micro-decode for notes', async () => {
  const definition = {
    initial: 'reflecting',
    states: {
      reflecting: {
        probe: {
          prompt: (ctx) => `Summarize finding at offset ${ctx.offset}:`,
          maxTokens: 16,
          op: 'save_note',
          target: 'idle',
        },
      },
      idle: {},
    },
  }

  const mockClient = {
    probe: async (prompt, candidates, maxTokens) => {
      assert.strictEqual(maxTokens, 16)
      assert.ok(prompt.includes('offset 512'))
      return { winningIdx: 0, confidence: 0, entropy: 0, text: 'File contains Vulkan compute headers.' }
    },
  }

  const ops = {
    save_note: (ctx, probeRes) => ({
      notes: [...(ctx.notes || []), probeRes.text],
    }),
  }

  const fsm = Afsm({
    definition,
    client: mockClient,
    initialContext: { offset: 512, notes: [] },
    ops,
  })

  const res = await fsm.step()
  assert.strictEqual(res.to, 'idle')
  assert.strictEqual(res.op, 'save_note')
  assert.strictEqual(fsm.getState().context.notes.length, 1)
  assert.strictEqual(
    fsm.getState().context.notes[0],
    'File contains Vulkan compute headers.',
  )
})

test('Afsm parses and runs foundational templates chat.yaml and reader.yaml', () => {
  const fs = require('node:fs')
  const path = require('node:path')
  const yaml = require('js-yaml')

  const chatRaw = fs.readFileSync(path.resolve(__dirname, '../templates/chat.yaml'), 'utf-8')
  const chatDef = yaml.load(chatRaw)
  assert.strictEqual(chatDef.name, 'chat')
  assert.strictEqual(chatDef.initial, 'idle')
  assert.ok(chatDef.states.idle.probe.candidates.length >= 2)

  const readerRaw = fs.readFileSync(path.resolve(__dirname, '../templates/reader.yaml'), 'utf-8')
  const readerDef = yaml.load(readerRaw)
  assert.strictEqual(readerDef.name, 'reader')
  assert.strictEqual(readerDef.initial, 'reading')
  assert.ok(readerDef.states.reading.probe.candidates.length >= 4)
})
