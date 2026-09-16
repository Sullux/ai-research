const fs = require('fs')
const path = require('path')
const yaml = require('js-yaml')
const { refs } = require('./state')
const {
  formatTurn1,
  formatUserTurn,
  formatUserDecisionTurn,
} = require('../../template')
const { Afsm } = require('../../afsm')

let chatTemplate = null
const getChatTemplate = () => {
  if (!chatTemplate) {
    try {
      const raw = fs.readFileSync(
        path.resolve(__dirname, '../../../templates/chat.yaml'),
        'utf-8',
      )
      chatTemplate = yaml.load(raw)
    } catch (_) {}
  }
  return chatTemplate
}

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
    refs.channelManager?.registerChannel({
      id: 'chat/user',
      path: savedMsg.relPath,
      type: 'push',
      cursor: 0,
      isFocused: true,
    })
    const chunk = refs.channelManager?.readChunk('chat/user', refs.vfs, 512)
    const readContent = chunk?.content || savedMsg.payload

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
    turnBody = readContent

    // Autonomic task triage & titling via Afsm (chat.yaml)
    const activeTasks = refs.taskManager?.getActiveTasks() || []
    const chatDef = getChatTemplate()
    if (activeTasks.length > 0 && refs.client && chatDef) {
      const triageSeq = notItem ? notItem.seq : parseInt(savedMsg.id, 10)
      const candList = activeTasks
        .map((t, idx) => `${idx + 1}: ${t.title}`)
        .join('\n')
      const candidates = activeTasks
        .map((t, idx) => ({
          key: String(idx + 1),
          op: 'steer_task',
          target: 'responding',
          task: t,
        }))
        .concat([{ key: '0', op: 'create_task', target: 'titling' }])

      const ops = {
        create_task: (fsmCtx) => {
          const firstLine = val.trim().split('\n')[0]
          const fallbackTitle =
            firstLine.length > 80 ? `${firstLine.slice(0, 77)}...` : firstLine
          const newTask = refs.taskManager.createTask(fallbackTitle)
          refs.store?.addStreamEntry({
            type: 'system',
            title: '🎯 NEW TASK',
            content: `Autonomic triage assigned Event not${triageSeq} to new Task #${newTask.id}: "${fallbackTitle}"`,
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
          return { taskId: newTask.id, valSample: val.slice(0, 256) }
        },
        set_title: (fsmCtx, probeRes) => {
          if (probeRes?.text && fsmCtx.taskId) {
            refs.taskManager.updateTask(fsmCtx.taskId, { title: probeRes.text })
            refs.store?.addStreamEntry({
              type: 'system',
              title: '🏷 TASK TITLE',
              content: `Task #${fsmCtx.taskId} titled: "${probeRes.text}"`,
            })
          }
        },
        steer_task: (fsmCtx, probeRes) => {
          const targetTask = activeTasks[probeRes.winningIdx]
          if (targetTask) {
            refs.taskManager.setActiveTask(targetTask.id)
            refs.store?.addStreamEntry({
              type: 'system',
              title: '🎯 ROUTED',
              content: `Autonomic triage routed Event not${triageSeq} to Task #${targetTask.id}`,
            })
          }
          refs.interruptedTurnNotificationId = null
        },
      }

      const fsm = Afsm({
        definition: { ...chatDef, initial: 'on_input' },
        client: refs.client,
        initialContext: {
          val,
          valSample: val.slice(0, 256),
          candList,
          candidates,
        },
        ops,
      })

      fsm.step().then((res) => {
        if (res.to === 'titling') {
          fsm.step()
        }
      })
    } else if (refs.taskManager) {
      const firstLine = val.trim().split('\n')[0]
      const fallbackTitle =
        firstLine.length > 80 ? `${firstLine.slice(0, 77)}...` : firstLine
      const newTask = refs.taskManager.createTask(fallbackTitle)
      if (refs.client && chatDef) {
        const ops = {
          set_title: (fsmCtx, probeRes) => {
            if (probeRes?.text && fsmCtx.taskId) {
              refs.taskManager.updateTask(fsmCtx.taskId, { title: probeRes.text })
              refs.store?.addStreamEntry({
                type: 'system',
                title: '🏷 TASK TITLE',
                content: `Task #${fsmCtx.taskId} titled: "${probeRes.text}"`,
              })
            }
          },
        }
        const fsm = Afsm({
          definition: { ...chatDef, initial: 'titling' },
          client: refs.client,
          initialContext: {
            taskId: newTask.id,
            valSample: val.slice(0, 256),
          },
          ops,
        })
        fsm.step()
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
