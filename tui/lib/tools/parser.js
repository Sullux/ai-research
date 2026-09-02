const { formatToolResponse } = require('../template')
const { SyntaxTracker } = require('../stream/syntax')

const TOOL_CALL_START = '<|tool_call>'
const TOOL_CALL_END = '<tool_call|>'

const parseToolCall = (str) => {
  const startIdx = str.indexOf(TOOL_CALL_START)
  if (startIdx === -1) return null
  const endIdx = str.indexOf(TOOL_CALL_END, startIdx)
  if (endIdx === -1) return null

  const raw = str.slice(startIdx + TOOL_CALL_START.length, endIdx).trim()
  const colonIdx = raw.indexOf(':')
  if (colonIdx === -1) return null

  const nameAndArgs = raw.slice(colonIdx + 1)
  const braceIdx = nameAndArgs.indexOf('{')
  if (braceIdx === -1) return { name: nameAndArgs.trim(), args: {} }

  const name = nameAndArgs.slice(0, braceIdx).trim()
  const argsStr = nameAndArgs.slice(braceIdx)
  try {
    return { name, args: JSON.parse(argsStr) }
  } catch (_) {
    try {
      const sanitized = argsStr.replace(/\r?\n/g, '\\n')
      return { name, args: JSON.parse(sanitized) }
    } catch (_) {
      return { name, args: {} }
    }
  }
}

const toolParserFactory = () => (registry, client, store) => {
  let streamAccumulator = ''
  const syntaxTracker = SyntaxTracker()

  const ingestChunk = async (chunk) => {
    syntaxTracker.ingestChunk(chunk)
    streamAccumulator += chunk
    const parsed = parseToolCall(streamAccumulator)
    if (!parsed) return

    const endIdx = streamAccumulator.indexOf(TOOL_CALL_END)
    streamAccumulator = streamAccumulator.slice(endIdx + TOOL_CALL_END.length)

    store?.addStreamEntry?.({
      type: 'tool_call',
      title: `🛠️ TOOL: ${parsed.name}`,
      content: JSON.stringify(parsed.args, null, 2),
    })

    const result = await registry.execute(parsed.name, parsed.args)

    if (parsed.name === 'ask_user') {
      const msg = parsed.args.message || parsed.args.prompt || parsed.args.brief || 'User action required'
      store?.addConversationMessage?.({
        sender: 'Assistant',
        text: msg,
        waitingUser: true,
      })
    }

    store?.addStreamEntry?.({
      type: 'tool_result',
      title: `📦 RESULT: ${parsed.name}`,
      content: JSON.stringify(result, null, 2),
    })

    const responsePayload = formatToolResponse(parsed.name, result)
    client.sendInput(responsePayload)
  }

  const reset = () => {
    streamAccumulator = ''
    syntaxTracker.reset()
  }

  return { ingestChunk, reset, syntaxTracker }
}

module.exports = {
  parseToolCall,
  toolParserFactory,
  ToolParser: toolParserFactory(),
}
