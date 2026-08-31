const { describe, it } = require('node:test')
const assert = require('node:assert')
const { stateStoreFactory } = require('../lib/ui/state')

describe('UI StateStore', () => {
  it('initializes with default conversation and stream items', () => {
    const StateStore = stateStoreFactory()
    const store = StateStore()
    assert.strictEqual(store.state.conversation.length, 1)
    assert.strictEqual(store.state.stream.length, 1)
    assert.strictEqual(store.state.mode, 'chat')
    assert.strictEqual(store.state.isEditMode, false)
  })

  it('accumulates and flushes active thoughts into stream', () => {
    const StateStore = stateStoreFactory()
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
    const StateStore = stateStoreFactory()
    const store = StateStore()
    store.appendActiveResponse('System update completed successfully.')
    store.flushActiveResponse()

    assert.strictEqual(store.state.conversation.length, 2)
    assert.strictEqual(store.state.conversation[1].text, 'System update completed successfully.')
    assert.strictEqual(store.state.stream.length, 2)
    assert.strictEqual(store.state.stream[1].type, 'response')
  })

  it('tracks prompt history accurately', () => {
    const StateStore = stateStoreFactory()
    const store = StateStore()
    store.pushHistory('First prompt')
    store.pushHistory('Second prompt')

    assert.strictEqual(store.state.history.length, 2)
    assert.strictEqual(store.state.history[0], 'First prompt')
    assert.strictEqual(store.state.history[1], 'Second prompt')
  })

  it('hydrates conversation and stream from historical stream items', () => {
    const persisted = []
    const StateStore = stateStoreFactory()
    const store = StateStore((item) => persisted.push(item))

    const historyItems = [
      { id: '1', type: 'user', content: 'What is the date?', time: 1000 },
      { id: '2', type: 'thought', content: 'Checking date format', time: 1001 },
      { id: '3', type: 'response', content: 'It is March 3rd.', time: 1002 },
      { id: '4', type: 'ask_user', content: 'Do you want to proceed?', time: 1003 },
    ]

    store.hydrateFromStream(historyItems)

    assert.strictEqual(store.state.stream.length, 4)
    assert.strictEqual(store.state.conversation.length, 3) // user, response, ask_user
    assert.strictEqual(store.state.conversation[0].sender, 'User')
    assert.strictEqual(store.state.conversation[0].text, 'What is the date?')
    assert.strictEqual(store.state.conversation[1].sender, 'Assistant')
    assert.strictEqual(store.state.conversation[1].text, 'It is March 3rd.')
    assert.strictEqual(store.state.conversation[2].waitingUser, true)
    assert.strictEqual(store.state.conversation[2].text, 'Do you want to proceed?')
    assert.strictEqual(persisted.length, 0) // hydration does not re-emit persistence
  })
})
