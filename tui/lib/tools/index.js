const toolRegistryFactory = () => (session, client, timers) => {
  const tools = {
    recall: async (args) => {
      const query = args.query || args.keywords || ''
      const topK = args.top_k || 5
      client.sendMemQuery(query, topK)
      return { status: 'query_submitted', query, topK }
    },

    terminal_write: async (args) => {
      const input = args.input || ''
      const watch = args.watch || 'none'
      const debounceMs = args.debounce_ms || 250

      session.write(input)

      if (watch === 'screen_change' || watch === 'completion') {
        session.setWatch((screen) => {
          client.sendInput(`\n<|terminal_screen|>\n${screen}\n<|end_terminal_screen|>\n`)
        })
      }
      return { status: 'written', bytes: input.length, watch }
    },

    terminal_read: async (args) => {
      const view = args.view || 'screen'
      if (view === 'screen') {
        return { view: 'screen', content: session.buffer.screenText() }
      }
      if (args.page_offset !== undefined) {
        return { view: 'scrollback', lines: session.buffer.pageSlice(args.page_offset) }
      }
      const start = args.line_start || 0
      const count = args.line_count || 30
      return { view: 'scrollback', lines: session.buffer.scrollbackSlice(start, count) }
    },

    terminal_search: async (args) => {
      const pattern = args.pattern || ''
      const maxResults = args.max_results || 10
      return { pattern, results: session.buffer.search(pattern, maxResults) }
    },

    terminal_key: async (args) => {
      session.sendKey(args.key)
      return { status: 'key_sent', key: args.key }
    },

    terminal_reset: async () => {
      session.reset()
      return { status: 'terminal_reset' }
    },

    set_timer: async (args) => {
      const seconds = Number(args.seconds) || 10
      const note = args.note || 'wake-up alarm'
      const res = timers.setTimer(seconds, note, ({ note: n }) => {
        client.sendInput(`\n<|notification|>TIMER: ${n}<|end_notification|>\n`)
      })
      return res
    },

    cancel_timer: async (args) => {
      return timers.cancelTimer(args.timer_id || args.timerId)
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
