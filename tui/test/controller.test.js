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

  it('renders smart 4-line live stream cards with conditional expand hint', () => {
    const store = StateStore()
    store.state.stream = []
    // Short item
    store.addStreamEntry({ type: 'thought', title: '💭 THOUGHT', content: 'Short thought' })
    // Long item (more than 3 lines)
    const longContent = 'Line 1 word word word\nLine 2 word word word\nLine 3 word word word\nLine 4 word word word\nLine 5 word word word'
    store.addStreamEntry({ type: 'response', title: '🤖 ASSISTANT', content: longContent })

    controller.init(store, null, null, null, null)

    // Unselected state (mode = chat)
    store.setMode('chat')
    let nodes = controller.getStreamNodes()
    assert.strictEqual(nodes.length, 2)
    const longSpansUnselected = nodes[1].inner[0].inner
    assert(longSpansUnselected.some((s) => s.text.includes('(↑ 2 more)')))
    assert(!longSpansUnselected.some((s) => s.text.includes('(x to expand)')))

    // Selected state (mode = stream, selectedIdx = 1)
    store.setMode('stream')
    store.state.selectedIdx.stream = 1
    nodes = controller.getStreamNodes()
    const longSpansSelected = nodes[1].inner[0].inner
    assert(longSpansSelected.some((s) => s.text.includes('(x to expand)')))

    // Expanded state
    store.toggleExpandStreamItem(1)
    nodes = controller.getStreamNodes()
    const longSpansExpanded = nodes[1].inner[0].inner
    assert(longSpansExpanded.some((s) => s.text.includes('(x to collapse)')))
  })

  it('copies selected conversation or stream item on c key', () => {
    const { getSelectedText, copyToClipboard } = require('../lib/ui/controller/clipboard')
    const store = StateStore()
    controller.init(store, null, null, null, null)

    // Chat mode copy
    store.state.mode = 'chat'
    store.state.conversation = [
      { id: '1', sender: 'User', text: 'First user prompt', time: 1000 },
      { id: '2', sender: 'Assistant', text: 'First assistant reply', time: 2000 },
    ]
    store.state.selectedIdx.chat = 1

    let written = ''
    const mockOut = {
      write: (data) => { written += data },
    }

    assert.strictEqual(getSelectedText(store.state), 'First assistant reply')
    copyToClipboard(getSelectedText(store.state), mockOut)
    assert.ok(written.startsWith('\x1b]52;c;'))
    assert.ok(written.endsWith('\x07'))
    const b64 = written.slice(7, -1)
    assert.strictEqual(Buffer.from(b64, 'base64').toString('utf-8'), 'First assistant reply')

    // Stream mode copy
    store.state.mode = 'stream'
    store.state.stream = [
      { id: 's1', type: 'thought', title: '💭 THOUGHT', content: 'Deep reasoning steps', time: 3000 },
    ]
    store.state.selectedIdx.stream = 0
    assert.strictEqual(getSelectedText(store.state), 'Deep reasoning steps')

    // Test key routing
    let keyHandled = false
    controller.onGlobalKey({ redraw: () => {} }, { key: 'c', stopPropagation: () => { keyHandled = true } })
    assert.strictEqual(keyHandled, true)
  })
})
