const MAX_STREAM_ITEMS = 1000
const MAX_CONVERSATION_ITEMS = 500

const stateStoreFactory = () => (onStreamItem) => {
  const state = {
    conversation: [
      {
        id: 'init',
        sender: 'Assistant',
        text: 'Cognitive Engine Ready. Press Enter to type a prompt or task.',
        time: Date.now(),
      },
    ],
    stream: [
      {
        id: 's-init',
        type: 'system',
        title: '⚙ SYSTEM',
        content: 'Vulkan RDNA 3.5 Engine connected. Task stack active.',
        time: Date.now(),
        expanded: false,
      },
    ],
    activeThought: '',
    activeThoughtTime: 0,
    activeResponse: '',
    activeResponseTime: 0,
    status: 'Idle | tok/s: 0.0 | Memory: 0 episodes',
    isGenerating: false,
    isPaused: false,
    mode: 'chat', // 'chat' | 'stream' | 'plan'
    isEditMode: false,
    overlay: null, // null | 'stream' | 'plan'
    dimensions: { cols: 100, rows: 30 },
    history: [],
    historyIdx: -1,
    cachedDraft: '',
    stickyScroll: { chat: true, stream: true },
    selectedIdx: { chat: 0, stream: 0, plan: 0 },
  }

  const addConversationMessage = (msg) => {
    const item = {
      id: msg.id || `msg-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      time: msg.time || Date.now(),
      ...msg,
    }
    state.conversation.push(item)
    if (state.conversation.length > MAX_CONVERSATION_ITEMS) {
      state.conversation.shift()
    }
    if (state.stickyScroll.chat) {
      state.selectedIdx.chat = state.conversation.length - 1
    }
  }

  const addStreamEntry = (entry, shouldPersist = true) => {
    const item = {
      id: entry.id || `str-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      time: entry.time || Date.now(),
      expanded: false,
      ...entry,
    }
    state.stream.push(item)
    if (state.stream.length > MAX_STREAM_ITEMS) {
      state.stream.shift()
    }
    if (state.stickyScroll.stream) {
      state.selectedIdx.stream = state.stream.length - 1
    }
    if (shouldPersist && onStreamItem) {
      onStreamItem(item)
    }
  }

  const hydrateFromStream = (items) => {
    if (!items || items.length === 0) return
    state.conversation = []
    state.stream = []

    for (const item of items) {
      addStreamEntry(item, false)
      if (item.type === 'user') {
        addConversationMessage({ sender: 'User', text: item.content, time: item.time, id: item.id })
      } else if (item.type === 'response') {
        addConversationMessage({ sender: 'Assistant', text: item.content, time: item.time, id: item.id })
      } else if (item.type === 'ask_user') {
        addConversationMessage({ sender: 'Assistant', text: item.content, time: item.time, id: item.id, waitingUser: true })
      }
    }
    if (state.conversation.length === 0) {
      addConversationMessage({ sender: 'Assistant', text: 'Cognitive Engine Ready. Press Enter to type a prompt or task.' })
    }
  }

  const appendActiveThought = (chunk) => {
    if (!state.activeThought) {
      state.activeThoughtTime = Date.now()
    }
    state.activeThought += chunk
  }

  const cleanThought = (raw) => {
    let t = (raw || '').trim()
    if (t.startsWith('thought')) {
      t = t.slice(7).trim()
    }
    return t
  }

  const flushActiveThought = () => {
    const content = cleanThought(state.activeThought)
    const time = state.activeThoughtTime || Date.now()
    state.activeThought = ''
    state.activeThoughtTime = 0
    if (!content) return
    addStreamEntry({
      type: 'thought',
      title: '💭 THOUGHT',
      content,
      time,
    })
  }

  const appendActiveResponse = (chunk) => {
    if (!state.activeResponse) {
      state.activeResponseTime = Date.now()
    }
    state.activeResponse += chunk
  }

  const flushActiveResponse = () => {
    if (!state.activeResponse) return
    const text = state.activeResponse
    const time = state.activeResponseTime || Date.now()
    state.activeResponse = ''
    state.activeResponseTime = 0
    addConversationMessage({
      sender: 'Assistant',
      text: text.trim(),
      time,
    })
    addStreamEntry({
      type: 'response',
      title: '🤖 ASSISTANT',
      content: text.trim(),
      time,
    })
  }

  const setStatus = (s) => { state.status = s }
  const setGenerating = (g) => { state.isGenerating = g }
  const setPaused = (p) => { state.isPaused = p }
  const setMode = (m) => { state.mode = m }
  const setEditMode = (e) => { state.isEditMode = e }
  const setOverlay = (o) => { state.overlay = o }
  const setDimensions = (cols, rows) => { state.dimensions = { cols, rows } }

  const toggleExpandStreamItem = (idx) => {
    const item = state.stream[idx]
    if (item) item.expanded = !item.expanded
  }

  const pushHistory = (txt) => {
    if (!txt || typeof txt !== 'string') return
    const trimmed = txt.trim()
    if (!trimmed) return
    state.history.push(trimmed)
    state.historyIdx = state.history.length
    state.cachedDraft = ''
  }

  return {
    state,
    addConversationMessage,
    addStreamEntry,
    hydrateFromStream,
    appendActiveThought,
    flushActiveThought,
    appendActiveResponse,
    flushActiveResponse,
    setStatus,
    setGenerating,
    setPaused,
    setMode,
    setEditMode,
    setOverlay,
    setDimensions,
    toggleExpandStreamItem,
    pushHistory,
  }
}

module.exports = {
  stateStoreFactory,
  StateStore: stateStoreFactory(),
}
