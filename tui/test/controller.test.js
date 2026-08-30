const { describe, it } = require('node:test')
const assert = require('node:assert')
const controller = require('../lib/ui/controller')
const { StateStore } = require('../lib/ui/state')
const { Orchestrator } = require('../lib/orchestrator')

describe('UI Controller & Layout Tiering', () => {
  it('computes layout tier based on terminal width', () => {
    assert.strictEqual(controller.getLayoutTier(220), 1)
    assert.strictEqual(controller.getLayoutTier(180), 2)
    assert.strictEqual(controller.getLayoutTier(120), 3)
  })

  it('detects clipped dimensions below minimum 80x30', () => {
    assert.strictEqual(controller.isClipped(70, 40), true)
    assert.strictEqual(controller.isClipped(100, 20), true)
    assert.strictEqual(controller.isClipped(100, 35), false)
  })

  it('generates conversation, stream, and plan nodes', () => {
    const store = StateStore()
    const orch = Orchestrator()
    orch.createPlan('Test Plan', ['Step A', 'Step B'])
    controller.init(store, null, null, orch, null)

    const convNodes = controller.getConversationNodes()
    assert(convNodes.length >= 1)

    const streamNodes = controller.getStreamNodes()
    assert(streamNodes.length >= 1)

    const planNodes = controller.getPlanNodes()
    assert(planNodes.length >= 2)
  })

  it('toggles mode on a, s, d hotkeys', () => {
    const store = StateStore()
    controller.init(store, null, null, null, null)

    let redrawn = false
    const mockCtx = {
      redraw: () => { redrawn = true },
      width: 150,
    }

    controller.onGlobalKey(mockCtx, { key: 's', stopPropagation: () => {} })
    assert.strictEqual(store.state.mode, 'stream')
    assert.strictEqual(store.state.overlay, 'stream')

    controller.onGlobalKey(mockCtx, { key: 'a', stopPropagation: () => {} })
    assert.strictEqual(store.state.mode, 'chat')
    assert.strictEqual(store.state.overlay, null)
  })

  it('injects system prompt on first turn input submit', () => {
    const store = StateStore()
    const orch = Orchestrator()
    let sentInput = ''
    const mockClient = {
      sendInput: (payload) => { sentInput = payload },
    }

    controller.init(store, mockClient, null, orch, null, 'Test System Prompt')

    const mockCtx = {
      redraw: () => {},
      setFocus: () => {},
    }

    controller.onSubmitInput(mockCtx, { value: 'Hello world', node: { value: 'Hello world', cursor: 11 } })
    assert.ok(sentInput.includes('<|turn>system\n<|think|>\nTest System Prompt\n<turn|>'))
    assert.ok(sentInput.includes('<|turn>user\nHello world\n<turn|>\n<|turn>model\n'))

    // Second turn should NOT repeat system prompt
    controller.onSubmitInput(mockCtx, { value: 'Follow up', node: { value: 'Follow up', cursor: 9 } })
    assert.ok(!sentInput.includes('<|turn>system'))
    assert.ok(sentInput.includes('<|turn>user\nFollow up\n<turn|>\n<|turn>model\n'))
  })
})
