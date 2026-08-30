const { describe, it } = require('node:test')
const assert = require('node:assert')
const { StateStore } = require('../lib/ui/state')

describe('UI StateStore', () => {
  it('initializes with default conversation and stream items', () => {
    const store = StateStore()
    assert.strictEqual(store.state.conversation.length, 1)
    assert.strictEqual(store.state.stream.length, 1)
    assert.strictEqual(store.state.mode, 'chat')
    assert.strictEqual(store.state.isEditMode, false)
  })

  it('accumulates and flushes active thoughts into stream', () => {
    const store = StateStore()
    store.appendActiveThought('Checking filesystem')
    store.appendActiveThought(' for package managers')
    assert.strictEqual(store.state.activeThought, 'Checking filesystem for package managers')

    store.flushActiveThought()
    assert.strictEqual(store.state.activeThought, '')
    assert.strictEqual(store.state.stream.length, 2)
    assert.strictEqual(store.state.stream[1].type, 'thought')
    assert.strictEqual(store.state.stream[1].content, 'Checking filesystem for package managers')
  })

  it('accumulates and flushes active response into conversation and stream', () => {
    const store = StateStore()
    store.appendActiveResponse('System update completed successfully.')
    store.flushActiveResponse()

    assert.strictEqual(store.state.conversation.length, 2)
    assert.strictEqual(store.state.conversation[1].text, 'System update completed successfully.')
    assert.strictEqual(store.state.stream.length, 2)
    assert.strictEqual(store.state.stream[1].type, 'response')
  })

  it('tracks prompt history accurately', () => {
    const store = StateStore()
    store.pushHistory('First prompt')
    store.pushHistory('Second prompt')

    assert.strictEqual(store.state.history.length, 2)
    assert.strictEqual(store.state.history[0], 'First prompt')
    assert.strictEqual(store.state.history[1], 'Second prompt')
  })
})
