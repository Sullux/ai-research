const TOOL_CALL_START = '<|tool_call|>'
const TOOL_CALL_END = '<tool_call|>'

const parseToolCall = (str) => {
  const startIdx = str.indexOf(TOOL_CALL_START)
  if (startIdx === -1) return null
  const endIdx = str.indexOf(TOOL_CALL_END, startIdx)
  if (endIdx === -1) return null

  const raw = str.slice(startIdx + TOOL_CALL_START.length, endIdx).trim()
  // format: "call:tool_name{...}"
  const colonIdx = raw.indexOf(':')
  if (colonIdx === -1) return null

  const nameAndArgs = raw.slice(colonIdx + 1)
  const braceIdx = nameAndArgs.indexOf('{')
  if (braceIdx === -1) return { name: nameAndArgs.trim(), args: {} }

  const name = nameAndArgs.slice(0, braceIdx).trim()
  const argsStr = nameAndArgs.slice(braceIdx)
  try {
    const args = JSON.parse(argsStr)
    return { name, args }
  } catch (_) {
    try {
      const sanitized = argsStr.replace(/\r?\n/g, '\\n')
      const args = JSON.parse(sanitized)
      return { name, args }
    } catch (_) {
      return { name, args: {} }
    }
  }
}

const toolParserFactory = () => (registry, client) => {
  let streamAccumulator = ''

  const ingestChunk = async (chunk) => {
    streamAccumulator += chunk
    const parsed = parseToolCall(streamAccumulator)
    if (!parsed) return

    // Clear accumulator up to tool call end
    const endIdx = streamAccumulator.indexOf(TOOL_CALL_END)
    streamAccumulator = streamAccumulator.slice(endIdx + TOOL_CALL_END.length)

    const result = await registry.execute(parsed.name, parsed.args)
    const responsePayload = `\n<|tool_response|>${JSON.stringify(result)}<tool_response|>\n`
    client.sendInput(responsePayload)
  }

  const reset = () => {
    streamAccumulator = ''
  }

  return { ingestChunk, reset }
}

module.exports = {
  parseToolCall,
  toolParserFactory,
  ToolParser: toolParserFactory(),
}
