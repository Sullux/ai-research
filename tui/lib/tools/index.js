const toolRegistryFactory = () => (session, client, timers, orchestrator) => {
  const tools = {
    plan: async (args) => {
      if (!orchestrator) return { error: 'No orchestrator available' }
      const plan = orchestrator.createPlan(args.brief, args.steps)
      client?.sendMemCommit?.()
      return {
        status: 'plan_created',
        id: plan.id,
        brief: plan.brief,
        steps: plan.steps.map((s) => ({ id: s.id, brief: s.brief })),
        active_step: plan.steps[0]?.id || null,
      }
    },

    done: async (args) => {
      if (!orchestrator) return { error: 'No orchestrator available' }
      const res = orchestrator.completeActiveStep(args.summary)
      if (!res) return { status: 'no_active_step' }
      return {
        status: 'step_completed',
        completed_step: res.completedStep.id,
        summary: res.completedStep.summary,
        next_step: res.nextStep?.id || null,
      }
    },

    defer: async (args) => {
      if (!orchestrator) return { error: 'No orchestrator available' }
      const duration = args.duration || '10s'
      const reason = args.reason || 'waiting'
      const res = orchestrator.deferActiveStep(duration, reason, (step) => {
        const thought = orchestrator.buildThoughtPrefix('TIMER_WAKE', { step })
        client?.sendInput?.(thought)
      })
      if (!res) return { status: 'no_active_step_to_defer' }
      return {
        status: 'step_deferred',
        step: res.deferredStep.id,
        duration: duration,
        reason: reason,
      }
    },

    ask_user: async (args) => {
      if (!orchestrator) return { error: 'No orchestrator available' }
      const brief = args.brief || 'user action required'
      const message = args.message || args.prompt || brief
      const res = orchestrator.askUserActiveStep(brief, message)
      return {
        status: 'waiting_for_user',
        step: res.step?.id || null,
        brief: res.step?.waitingForUser?.brief || brief,
        message: message,
      }
    },

    recall: async (args) => {
      const query = args.query || args.keywords || ''
      const topK = args.top_k || 5
      client?.sendMemQuery?.(query, topK)
      return { status: 'query_submitted', query, topK }
    },

    terminal_write: async (args) => {
      const input = args.input || ''
      const watch = args.watch || 'none'

      session?.write?.(input)

      if (watch === 'screen_change' || watch === 'completion') {
        session?.setWatch?.((screen) => {
          client?.sendInput?.(`\n<|terminal_screen|>\n${screen}\n<|end_terminal_screen|>\n`)
        })
      }
      return { status: 'written', bytes: input.length, watch }
    },

    terminal_read: async (args) => {
      const view = args.view || 'screen'
      if (view === 'screen') {
        return { view: 'screen', content: session?.buffer?.screenText?.() || '' }
      }
      if (args.page_offset !== undefined) {
        return { view: 'scrollback', lines: session?.buffer?.pageSlice?.(args.page_offset) || [] }
      }
      const start = args.line_start || 0
      const count = args.line_count || 30
      return { view: 'scrollback', lines: session?.buffer?.scrollbackSlice?.(start, count) || [] }
    },

    terminal_search: async (args) => {
      const pattern = args.pattern || ''
      const maxResults = args.max_results || 10
      return { pattern, results: session?.buffer?.search?.(pattern, maxResults) || [] }
    },

    terminal_key: async (args) => {
      session?.sendKey?.(args.key)
      return { status: 'key_sent', key: args.key }
    },

    terminal_reset: async () => {
      session?.reset?.()
      return { status: 'terminal_reset' }
    },
  }

  const execute = async (name, args) => {
    const fn = tools[name]
    if (!fn) return { error: `Unknown tool: ${name}` }
    try {
      return await fn(args)
    } catch (err) {
      return { error: err.message }
    }
  }

  return { tools, execute }
}

module.exports = {
  toolRegistryFactory,
  ToolRegistry: toolRegistryFactory(),
}
