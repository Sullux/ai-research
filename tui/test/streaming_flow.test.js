const { describe, it } = require('node:test')
const assert = require('node:assert')
const { SyntaxTracker } = require('../lib/stream/syntax')
const { stateStoreFactory } = require('../lib/ui/state')
const controller = require('../lib/ui/controller')

describe('Streaming Transduction Flow & In-Flight Barge-In', () => {
  it('detects natural resting points across multiline streamed tokens', () => {
    const tracker = SyntaxTracker()

    // 1. Initial code block starts
    tracker.ingestChunk('```typescript\n')
    assert.strictEqual(tracker.isAtRest(), false)
    assert.strictEqual(tracker.isNaturalBoundary('```typescript\n'), false)

    // 2. Code inside block
    tracker.ingestChunk('const x = (10 + 20);\n')
    assert.strictEqual(tracker.isAtRest(), false)

    // 3. Code block closes
    tracker.ingestChunk('```\n\n')
    assert.strictEqual(tracker.isAtRest(), true)
    assert.strictEqual(tracker.isNaturalBoundary('```typescript\nconst x = (10 + 20);\n```\n\n'), true)
  })

  it('handles in-flight user barge-in cleanly without losing active thoughts or responses', () => {
    const StateStore = stateStoreFactory()
    const store = StateStore()

    let sentPayloads = []
    const mockClient = {
      sendInput: (payload) => { sentPayloads.push(payload) },
    }
    const mockSession = {}
    const mockOrchestrator = {
      getWaitingForUserTasks: () => [],
      getActiveStep: () => null,
    }
    const mockTimers = {}

    controller.init(store, mockClient, mockSession, mockOrchestrator, mockTimers, 'You are Gemma.')

    // Simulate model actively generating a thought and partial response
    store.appendActiveThought('Analyzing performance data...')
    store.appendActiveResponse('Based on the initial benchmarks, ')
    store.setGenerating(true)

    // User barges in with new input
    const mockCtx = {
      setFocus: () => {},
      redraw: () => {},
    }
    controller.onSubmitInput(mockCtx, {
      value: 'Wait, focus on memory bandwidth instead.',
    })

    // Verify in-flight thought was flushed
    const thoughts = store.state.stream.filter(s => s.type === 'thought')
    assert.strictEqual(thoughts.length, 1)
    assert.strictEqual(thoughts[0].content, 'Analyzing performance data...')

    // Verify in-flight response was flushed to conversation before user turn
    const conv = store.state.conversation
    assert.strictEqual(conv.length >= 3, true) // init + partial assistant + user
    const partialAssistant = conv.find(c => c.sender === 'Assistant' && c.text.includes('Based on the initial'))
    assert.ok(partialAssistant)

    // Verify user message appended after partial assistant message
    const userMsg = conv[conv.length - 1]
    assert.strictEqual(userMsg.sender === 'User', true)
    assert.strictEqual(userMsg.text, 'Wait, focus on memory bandwidth instead.')

    // Verify payload sent to backend
    assert.strictEqual(sentPayloads.length, 1)
    assert.strictEqual(sentPayloads[0].includes('Wait, focus on memory bandwidth instead.'), true)
  })
})
