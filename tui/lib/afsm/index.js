const afsmFactory =
  () =>
  ({ definition, client, initialContext = {}, ops = {}, onTransition }) => {
    let currentState = definition.initial || Object.keys(definition.states)[0]
    let context = { ...initialContext }

    const getState = () => ({
      currentState,
      context: { ...context },
    })

    const setContext = (updates) => {
      context = { ...context, ...updates }
      return { ...context }
    }

    const step = async (overridePrompt) => {
      const stateDef = definition.states?.[currentState]
      if (!stateDef) throw new Error(`Unknown A-FSM state: ${currentState}`)
      const probeDef = stateDef.probe
      if (!probeDef) return { state: currentState, context: { ...context } }

      const rawPrompt = overridePrompt || probeDef.prompt
      const prompt =
        typeof rawPrompt === 'function'
          ? rawPrompt(context)
          : typeof rawPrompt === 'string'
          ? rawPrompt.replace(/\{\{(\w+)\}\}/g, (_, k) => context[k] ?? '')
          : rawPrompt

      const resolvedCands =
        typeof probeDef.candidates === 'function'
          ? probeDef.candidates(context)
          : probeDef.candidates === 'context' || probeDef.candidates === true
          ? context.candidates || []
          : probeDef.candidates || []
      const candEntries = Array.isArray(resolvedCands)
        ? resolvedCands.map((c) => [c.key || c.id || String(c), c])
        : Object.entries(resolvedCands)
      const candidates = candEntries.map(([k]) => k)
      const maxDecodeTokens =
        probeDef.maxTokens || (candidates.length > 0 ? 1 : 14)
      const minCertainty = probeDef.minCertainty || 0.6

      const probeRes = await client.probe(
        prompt,
        candidates,
        maxDecodeTokens,
        minCertainty,
      )

      let selectedOp = probeDef.op
      let nextState = probeDef.target || currentState

      if (candEntries.length > 0 && probeRes) {
        const winningEntry = candEntries[probeRes.winningIdx]
        const candConfig = winningEntry ? winningEntry[1] : null
        if (candConfig) {
          selectedOp = candConfig.op || selectedOp
          nextState = candConfig.target || nextState
        }
      }

      if (selectedOp && typeof ops[selectedOp] === 'function') {
        const opResult = await ops[selectedOp](context, probeRes)
        if (opResult && typeof opResult === 'object') {
          context = { ...context, ...opResult }
        }
      }

      const prev = currentState
      currentState = nextState

      if (typeof onTransition === 'function') {
        onTransition({ from: prev, to: currentState, op: selectedOp, probeRes, context })
      }

      return {
        from: prev,
        to: currentState,
        op: selectedOp,
        probeRes,
        context: { ...context },
      }
    }

    return {
      getState,
      setContext,
      step,
    }
  }

module.exports = {
  afsmFactory,
  Afsm: afsmFactory(),
}
