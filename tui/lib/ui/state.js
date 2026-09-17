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
    activeThoughtExpanded: false,
    activeResponse: '',
    activeResponseTime: 0,
    activeResponseExpanded: false,
    pendingInterjection: null,
    status: '[Engine] Initializing GPU compute & pre-caching working state...',
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
    if (state.conversation.length === 0 || item.time >= (state.conversation[state.conversation.length - 1].time || 0)) {
      state.conversation.push(item)
    } else {
      let idx = state.conversation.length - 1
      while (idx >= 0 && (state.conversation[idx].time || 0) > item.time) {
        idx--
      }
      state.conversation.splice(idx + 1, 0, item)
    }
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
    if (state.stream.length === 0 || item.time >= (state.stream[state.stream.length - 1].time || 0)) {
      state.stream.push(item)
    } else {
      let idx = state.stream.length - 1
      while (idx >= 0 && (state.stream[idx].time || 0) > item.time) {
        idx--
      }
      state.stream.splice(idx + 1, 0, item)
    }
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

    const sortedItems = [...items].sort((a, b) => (a.time || 0) - (b.time || 0))
    for (const item of sortedItems) {
      addStreamEntry(item, false)
    }

    // Populate conversation sorted by timestamp to preserve causal in-flight order
    const convItems = sortedItems.filter(it => ['user', 'response', 'ask_user'].includes(it.type))
    for (const item of convItems) {
      if (item.type === 'user') {
        const userText = item.rawText || (item.content ? item.content.replace(/^\[Event: [^\]]+ \| Source: [^\]]+\]\r?\n/, '') : '')
        addConversationMessage({ sender: 'User', text: userText, time: item.time, id: item.id })
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
    if (state.stickyScroll.stream) {
      state.selectedIdx.stream = state.stream.length
    }
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
    state.activeThoughtExpanded = false
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
    if (state.stickyScroll.chat) {
      state.selectedIdx.chat = state.conversation.length
    }
    if (state.stickyScroll.stream) {
      const th = cleanThought(state.activeThought)
      state.selectedIdx.stream = state.stream.length + (th ? 1 : 0)
    }
  }

  const setPendingInterjection = (msg, streamEntry) => {
    state.pendingInterjection = {
      ...msg,
      streamEntry,
    }
  }

  const flushPendingInterjection = () => {
    if (!state.pendingInterjection) return
    const { streamEntry, ...msg } = state.pendingInterjection
    state.pendingInterjection = null
    addConversationMessage(msg)
    if (streamEntry) addStreamEntry(streamEntry)
  }

  const clearActiveResponse = () => {
    state.activeResponse = ''
    state.activeResponseTime = 0
    state.activeResponseExpanded = false
  }

  const flushActiveResponse = () => {
    if (state.activeResponse) {
      const text = state.activeResponse
      const time = state.activeResponseTime || Date.now()
      state.activeResponse = ''
      state.activeResponseTime = 0
      state.activeResponseExpanded = false
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
    // If an in-flight user interjection was staged, insert it into conversation now
    flushPendingInterjection()
  }

  const setStatus = (s) => { state.status = s }
  const setGenerating = (g) => { state.isGenerating = g }
  const setPaused = (p) => { state.isPaused = p }
  const setMode = (m) => { state.mode = m }
  const setEditMode = (e) => { state.isEditMode = e }
  const setOverlay = (o) => { state.overlay = o }
  const setDimensions = (cols, rows) => { state.dimensions = { cols, rows } }

  const toggleExpandStreamItem = (idx) => {
    const liveThoughtIdx = cleanThought(state.activeThought) ? state.stream.length : -1
    const liveRespIdx = state.activeResponse ? (state.stream.length + (liveThoughtIdx !== -1 ? 1 : 0)) : -1

    if (idx === liveThoughtIdx && liveThoughtIdx !== -1) {
      state.activeThoughtExpanded = !state.activeThoughtExpanded
      return
    }
    if (idx === liveRespIdx && liveRespIdx !== -1) {
      state.activeResponseExpanded = !state.activeResponseExpanded
      return
    }
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
    clearActiveResponse,
    flushActiveResponse,
    setStatus,
    setGenerating,
    setPaused,
    setPendingInterjection,
    flushPendingInterjection,
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
