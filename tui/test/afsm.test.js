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

test('Afsm runs chat.yaml with dynamic context candidates and cascades to titling', async () => {
  const fs = require('node:fs')
  const path = require('node:path')
  const yaml = require('js-yaml')

  const chatRaw = fs.readFileSync(
    path.resolve(__dirname, '../templates/chat.yaml'),
    'utf-8',
  )
  const chatDef = yaml.load(chatRaw)

  const activeTasks = [{ id: 1, title: 'Check GPU memory' }]
  const candidates = [
    { key: '1', op: 'steer_task', target: 'responding', task: activeTasks[0] },
    { key: '0', op: 'create_task', target: 'titling' },
  ]

  const mockClient = {
    probe: async (prompt, cands, maxTokens) => {
      if (maxTokens === 14) {
        return { winningIdx: 0, confidence: 0, entropy: 0, text: 'Summarize 10GB File' }
      }
      return { winningIdx: 1, confidence: 0.99, entropy: 0.01, text: '' } // selects '0'
    },
  }

  let createdTask = null
  let titledTask = null

  const ops = {
    create_task: (ctx) => {
      createdTask = { id: 2, title: 'New Task' }
      return { taskId: 2, valSample: ctx.val.slice(0, 50) }
    },
    set_title: (ctx, probeRes) => {
      titledTask = { id: ctx.taskId, title: probeRes.text }
      return { title: probeRes.text }
    },
  }

  const fsm = Afsm({
    definition: chatDef,
    client: mockClient,
    initialContext: {
      val: 'How do I summarize a 10GB file using streaming?',
      valSample: 'How do I summarize a 10GB file using streaming?',
      candList: '1: Check GPU memory',
      candidates,
    },
    ops,
  })

  // Start in on_input
  fsm.setContext({ currentState: 'on_input' })
  // Force initial state
  fsm.reset = (st) => {
    // Afsm starts at initial: idle, we can call step with override state or pass initial in definition
  }

  const customDef = { ...chatDef, initial: 'on_input' }
  const fsmInput = Afsm({
    definition: customDef,
    client: mockClient,
    initialContext: {
      val: 'How do I summarize a 10GB file using streaming?',
      valSample: 'How do I summarize a 10GB file using streaming?',
      candList: '1: Check GPU memory',
      candidates,
    },
    ops,
  })

  const res1 = await fsmInput.step()
  assert.strictEqual(res1.from, 'on_input')
  assert.strictEqual(res1.to, 'titling')
  assert.strictEqual(res1.op, 'create_task')
  assert.strictEqual(createdTask.id, 2)

  const res2 = await fsmInput.step()
  assert.strictEqual(res2.from, 'titling')
  assert.strictEqual(res2.to, 'responding')
  assert.strictEqual(res2.op, 'set_title')
  assert.strictEqual(titledTask.title, 'Summarize 10GB File')
})

test('Afsm drives focus arbitration between channels', async () => {
  const fs = require('node:fs')
  const path = require('node:path')
  const yaml = require('js-yaml')
  const { ChannelManager } = require('../lib/channels')

  const focusRaw = fs.readFileSync(
    path.resolve(__dirname, '../templates/focus.yaml'),
    'utf-8',
  )
  const focusDef = yaml.load(focusRaw)
  const cm = ChannelManager()
  cm.registerChannel({ id: 'trm/build', isFocused: false })

  const mockClient = {
    probe: async () => ({
      winningIdx: 1, // selects '1': switch_focus
      confidence: 0.92,
      entropy: 0.08,
      text: '',
    }),
  }

  const ops = {
    switch_focus: (ctx) => {
      cm.setFocus(ctx.incomingChannel)
      return { focusedChannel: ctx.incomingChannel }
    },
    bookmark_interrupt: (ctx) => ({ bookmarked: true }),
  }

  const fsm = Afsm({
    definition: focusDef,
    client: mockClient,
    initialContext: {
      incomingChannel: 'trm/build',
      focusedChannel: cm.getFocused().id,
      preview: 'build error in shader',
    },
    ops,
  })

  assert.strictEqual(cm.getFocused().id, 'chat/user')
  const res = await fsm.step()
  assert.strictEqual(res.op, 'switch_focus')
  assert.strictEqual(cm.getFocused().id, 'trm/build')
  assert.strictEqual(fsm.getState().context.focusedChannel, 'trm/build')
})
