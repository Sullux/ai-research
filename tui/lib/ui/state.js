const stateStoreFactory = () => () => {
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
    activeResponse: '',
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
    state.conversation.push({
      id: `msg-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      time: Date.now(),
      ...msg,
    })
    if (state.stickyScroll.chat) {
      state.selectedIdx.chat = state.conversation.length - 1
    }
  }

  const addStreamEntry = (entry) => {
    state.stream.push({
      id: `str-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      time: Date.now(),
      expanded: false,
      ...entry,
    })
    if (state.stickyScroll.stream) {
      state.selectedIdx.stream = state.stream.length - 1
    }
  }

  const appendActiveThought = (chunk) => {
    state.activeThought += chunk
  }

  const flushActiveThought = () => {
    if (!state.activeThought) return
    addStreamEntry({
      type: 'thought',
      title: '💭 THOUGHT',
      content: state.activeThought,
    })
    state.activeThought = ''
  }

  const appendActiveResponse = (chunk) => {
    state.activeResponse += chunk
  }

  const flushActiveResponse = () => {
    if (!state.activeResponse) return
    const text = state.activeResponse
    state.activeResponse = ''
    addConversationMessage({
      sender: 'Assistant',
      text: text.trim(),
    })
    addStreamEntry({
      type: 'response',
      title: '🤖 ASSISTANT',
      content: text.trim(),
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
