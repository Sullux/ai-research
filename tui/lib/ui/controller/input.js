const { refs } = require('./state')
const {
  formatTurn1,
  formatUserTurn,
  formatUserDecisionTurn,
} = require('../../template')

const onSubmitInput = (ctx, payload) => {
  const val = payload.value?.trim()
  if (payload.node) {
    payload.node.value = ''
    payload.node.cursor = 0
  }
  if (!val || !refs.client) return

  // Flush in-flight thought before recording barge-in user turn.
  // We do NOT flush activeResponse here if it's currently generating tokens,
  // allowing the ongoing micro-burst to complete and place the interjection in causal order.
  if (refs.store?.state?.activeThought) {
    refs.store.flushActiveThought()
  }

  // If currently generating an assistant response, stage this message as pendingInterjection
  // so the conversation view preserves causal sequence (Response 1 -> Interjection -> Response 2).
  const isGeneratingResponse = Boolean(refs.store?.state?.activeResponse)
  const isGenerating = Boolean(
    refs.store?.state?.isGenerating ||
    refs.store?.state?.activeResponse ||
    refs.store?.state?.activeThought
  )

  // VFS message persistence & notification generation
  let turnHeader = ''
  let turnBody = val
  let savedMsg = null
  let eventId = null
  if (refs.vfs) {
    savedMsg = refs.vfs.saveUserMessage(val)

    const notItem = refs.notManager?.notify(
      savedMsg.relPath,
      savedMsg.preview,
      savedMsg.id,
      { isTurnContext: true, payload: savedMsg.payload },
    )
    if (isGenerating && refs.activeTurnNotificationId) {
      refs.interruptedTurnNotificationId = refs.activeTurnNotificationId
    }
    if (!isGenerating && notItem) {
      refs.notManager?.markServicing(notItem.id)
      refs.activeTurnNotificationId = notItem.id
    }
    eventId = notItem?.id || savedMsg.id
    turnHeader = `[Event: ${eventId} | Source: ${savedMsg.relPath}]\n`
    turnBody = savedMsg.payload

    // Autonomic task triage & titling via client.probe()
    const activeTasks = refs.taskManager?.getActiveTasks() || []
    if (activeTasks.length > 0 && refs.client) {
      const triageSeq = notItem ? notItem.seq : parseInt(savedMsg.id, 10)
      const candList = activeTasks
        .map((t, idx) => `${idx + 1}: ${t.title}`)
        .join('\n')
      const candDigits = activeTasks
        .map((_, idx) => String(idx + 1))
        .concat(['0'])
      const triagePrompt = `Classify whether the incoming user message is a constraint, follow-up, or steering for an active task, or a new independent task.\n\nActive tasks:\n${candList}\n0: New independent task\n\nIncoming message: "${val}"\n\nAnswer with only the index number:`
      refs.client.probe(triagePrompt, candDigits, 1).then((res) => {
        const winningIdx = res.winningIdx
        const isNewTask = winningIdx >= activeTasks.length
        if (isNewTask) {
          const firstLine = val.trim().split('\n')[0]
          const fallbackTitle =
            firstLine.length > 80 ? `${firstLine.slice(0, 77)}...` : firstLine
          const newTask = refs.taskManager.createTask(fallbackTitle)
          refs.store?.addStreamEntry({
            type: 'system',
            title: '🎯 NEW TASK',
            content: `Autonomic triage assigned Event not${triageSeq} to new Task #${newTask.id}: "${fallbackTitle}"`,
          })
          const titlePrompt = `Provide a concise 2 to 5 word title for this user request:\n"${val.slice(0, 256)}"\nOutput only the title, nothing else.`
          refs.client.probe(titlePrompt, [], 14).then((tRes) => {
            if (tRes?.text) {
              refs.taskManager.updateTask(newTask.id, { title: tRes.text })
              refs.store?.addStreamEntry({
                type: 'system',
                title: '🏷 TASK TITLE',
                content: `Task #${newTask.id} titled: "${tRes.text}"`,
              })
            }
          })
          const targetToSuspend =
            refs.interruptedTurnNotificationId ||
            (refs.activeTurnNotificationId !== notItem?.id
              ? refs.activeTurnNotificationId
              : null)
          if (targetToSuspend) {
            refs.notManager?.suspend(targetToSuspend)
          }
          refs.interruptedTurnNotificationId = null
          if (refs.activeTurnNotificationId === targetToSuspend) {
            refs.activeTurnNotificationId = null
          }
        } else {
          const targetTask = activeTasks[winningIdx]
          refs.taskManager.setActiveTask(targetTask.id)
          refs.store?.addStreamEntry({
            type: 'system',
            title: '🎯 ROUTED',
            content: `Autonomic triage routed Event not${triageSeq} to Task #${targetTask.id}`,
          })
          refs.interruptedTurnNotificationId = null
        }
      })
    } else if (refs.taskManager) {
      const firstLine = val.trim().split('\n')[0]
      const fallbackTitle =
        firstLine.length > 80 ? `${firstLine.slice(0, 77)}...` : firstLine
      const newTask = refs.taskManager.createTask(fallbackTitle)
      if (refs.client) {
        const titlePrompt = `Provide a concise 2 to 5 word title for this user request:\n"${val.slice(0, 256)}"\nOutput only the title, nothing else.`
        refs.client.probe(titlePrompt, [], 14).then((tRes) => {
          if (tRes?.text) {
            refs.taskManager.updateTask(newTask.id, { title: tRes.text })
            refs.store?.addStreamEntry({
              type: 'system',
              title: '🏷 TASK TITLE',
              content: `Task #${newTask.id} titled: "${tRes.text}"`,
            })
          }
        })
      }
    }

    if (!refs.taskManager && refs.activeTurnNotificationId && isGenerating) {
      refs.notManager?.suspend(refs.activeTurnNotificationId)
      refs.activeTurnNotificationId = null
    }
  }

  const turnContent = `${turnHeader}${turnBody}`
  refs.store?.pushHistory(val)

  if (isGeneratingResponse) {
    refs.store?.setPendingInterjection({ sender: 'User', text: val, time: Date.now() })
  } else if (!refs.isEngineReady) {
    refs.store?.addConversationMessage({ sender: 'User', text: val, waitingEngine: true })
  } else {
    refs.store?.addConversationMessage({ sender: 'User', text: val })
  }
  refs.store?.addStreamEntry({
    type: 'user',
    title: '👤 USER',
    content: turnContent,
    rawText: val,
  })

  refs.store?.setEditMode(false)
  ctx.setFocus?.(null)

  // Prepend any pending alerts rollup
  const alertsRollup = refs.notManager?.formatTurnAlerts?.() || ''

  let payloadText = ''
  if (!refs.hasSentFirstTurn && refs.systemPrompt) {
    refs.hasSentFirstTurn = true
    payloadText = formatTurn1(refs.systemPrompt, `${alertsRollup}${turnContent}`)
  } else {
    const waitingTasks = refs.orchestrator?.getWaitingForUserTasks?.() || []
    payloadText = waitingTasks.length > 0
      ? formatUserDecisionTurn(`${alertsRollup}${turnContent}`, waitingTasks)
      : formatUserTurn(`${alertsRollup}${turnContent}`)
  }

  refs.store?.setGenerating(true)
  if (!refs.isEngineReady) {
    // Engine still loading weights or restoring snapshot: stage turn to dispatch immediately upon ready
    refs.pendingInputTurn = payloadText
  } else if (!isGenerating || !refs.notManager) {
    refs.client.sendInput(payloadText)
  }
  ctx.redraw()
}

module.exports = {
  onSubmitInput,
}
