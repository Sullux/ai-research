const stateStoreFactory = () => () => {
  const state = {
    thoughts: '',
    dialogue: 'Cognitive Engine Ready. Type your prompt below.\n',
    terminal: '',
    status: 'Idle | tok/s: 0.0 | Memory: 0 episodes',
    input: '',
    isGenerating: false,
    history: [],
  }

  const setThoughts = (t) => { state.thoughts = t }
  const appendThought = (chunk) => { state.thoughts += chunk }
  const clearThoughts = () => { state.thoughts = '' }

  const appendDialogue = (chunk) => { state.dialogue += chunk }
  const setTerminal = (t) => { state.terminal = t }
  const setStatus = (s) => { state.status = s }
  const setInput = (i) => { state.input = i }
  const setGenerating = (g) => { state.isGenerating = g }

  return {
    state,
    setThoughts,
    appendThought,
    clearThoughts,
    appendDialogue,
    setTerminal,
    setStatus,
    setInput,
    setGenerating,
  }
}

module.exports = {
  stateStoreFactory,
  StateStore: stateStoreFactory(),
}
