# Client & Agent Runtime Architecture

## 1. Executive Summary & Design Principles

Traditional LLM agent runtimes interact with operating environments through **monolithic synchronous tool calls** and **unilateral context pushes**. When a user types or pastes a large document, or when a shell command produces megabytes of log output, the entire payload is injected directly into the active prompt turn.

In a continuous streaming architecture with a bounded physical KV cache (such as Channel's 4,096-slot dynamic ring buffer), this pattern causes catastrophic failures:
* **Context Shockwaves:** Pasting a 3,000-token file obliterates the model's active working memory, forcing immediate eviction of macro-reasoning anchors.
* **Interactive Latency Tax:** Forcing trivial inputs or fast commands (`ls`, `pwd`) through multi-turn paging and heavy JSON tool wrappers introduces hundreds of milliseconds of unnecessary round-trip delay.
* **Attentional Blindness:** Models either get derailed by unprompted context injections or completely ignore background processes while trapped in deep deliberative loops.
* **Zombie Process Leaks:** Spawning captive terminal sessions for simple one-off commands leaks pseudo-terminals (PTYs), processes, and disk artifacts.

The **Channel Client Runtime** solves these problems by treating the host environment as a real **Unix-Style Virtual File Subsystem (VFS)** coupled with an asynchronous interrupt controller.

---

## 2. The Virtual File Subsystem (VFS)

Rather than emulating a synthetic, in-memory virtual filesystem, the host leverages real OS primitives (directories, file permissions, append-only logs, and named FIFOs) rooted at a configurable path (`filesystemRoot` in `config.json`):

```
.agent/
├── msg/
│   ├── user/
│   │   ├── 1001.txt              <-- chmod 0444 (Read-Only)
│   │   └── 1002.txt              <-- chmod 0444 (Read-Only)
│   └── assistant/
│       └── 1002_reply.txt        <-- Generated replies
├── trm/
│   └── trm_server/
│       ├── screen.txt            <-- Live 24x80 rendered text grid
│       ├── stdout.log            <-- Raw append-only historical output stream
│       └── stdin                 <-- Named FIFO for piping interactive input
├── tmp/
│   ├── cmd_101.stdout.log        <-- Ephemeral log for background subshell
│   └── cmd_102.stdout.log
└── notify/
    ├── pending/
    │   └── not47                 <-- Active interrupt descriptor
    └── snoozed/
        └── not48                 <-- Suppressed until timer wake
```

### OS Security & Access Guarantees
* **Immutability of Inbound Messages:** Inbound user turns written to `msg/user/` are explicitly marked read-only via `chmod 0444`. Any attempt by the model or subshell to overwrite previous user input fails with `EACCES` (`Permission denied`).
* **Root-Relative Paths:** All paths in tool contracts and prompts (`msg/user/1001.txt`, `tmp/cmd_101.stdout.log`) are root-relative to `filesystemRoot`.
* **Bounded Reading (`vfs.read`):** Reading is strictly capped at **512 characters** (~128 tokens) per call, enforcing incremental, bite-sized consumption.

---

## 3. Subshell & Terminal Execution

Channel separates process execution into two distinct modalities:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      PROCESS EXECUTION MODALITIES                      │
├───────────────────────────────────┬────────────────────────────────────┤
│ 1. SUBSHELL EXECUTION (`cmd`)     │ 2. PERSISTENT TERMINALS (`trm`)   │
├───────────────────────────────────┼────────────────────────────────────┤
│ • Fast, one-off commands          │ • Interactive, stateful CLI tools  │
│ • Fast path: ≤ 250ms & ≤ 128 chars│ • 24x80 rendered `screen.txt` grid │
│   returns inline immediately      │ • Background PTY child process     │
│ • Detached path: > 250ms notifies │ • Real-time ANSI escape parsing    │
│   via interrupt when finished     │ • Control keys (`ctrl+c`, `enter`) │
└───────────────────────────────────┴────────────────────────────────────┘
```

### 3.1 Asynchronous Subshell (`cmd`)
* **Fast-Path Inline:** If a command finishes within **250 milliseconds** and outputs $\le 128$ characters, the result is returned directly inline to the model, eliminating unnecessary context jumps.
* **Spillover Logging:** If output exceeds 128 characters, the preview is truncated and full output is written to `tmp/cmd_<seq>.stdout.log`, nudging the model to inspect specific offsets via `read`.
* **Async Detached:** If a command runs longer than 250ms (e.g. `make`, `npm test`, `git clone`), the host detaches the process, returns a tracking ID (`cmd_<seq>`), and generates a notification interrupt when the process terminates.

### 3.2 Persistent Terminal Sessions (`trm`)
For interactive tools (e.g. `top`, `vim`, interactive debuggers, or REPLs), the host maintains persistent PTY sessions (`trm_open`, `trm_write`, `key`):
* Tracks a virtual 24-row by 80-column screen grid (`screen.txt`).
* Strips ANSI escape sequences and interprets cursor movements, scrolling, and carriage returns.
* Exposes control keys via `key({ session: "dev", char: "ctrl+c" })`.

---

## 4. The LIFO Interrupt Controller

All background events, subshell completions, timers, and user barge-ins enter the system as **Notifications**:

```
[ Inbound Event: User barge-in, cmd completion, timer wake ]
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ NotificationManager (tui/lib/notify/)                       │
│ - Generates uniform typed IDs: not101, not102 (no _)       │
│ - Enqueues in strict LIFO (Last-In, First-Out) priority     │
│ - Tracks state: PENDING -> SERVICING -> ACKNOWLEDGED        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
              [ Interrupt Dispatcher ]
              ├── ack({ id }): Permanently dismisses interrupt
              └── snooze({ id }): Suppresses until next turn or timer
```

* **Uniform Typed IDs:** Uses clean IDs without underscores (`not101`, `not102`) to prevent delimiter stutter and token hallucination in smaller models.
* **LIFO Priority:** The most recent interrupt takes immediate priority, allowing users to steer active execution without waiting for background queues to drain.
* **Auto-ACK on Resolution:** Completed turn-context notifications are automatically acknowledged at `STOP_END_OF_TURN`.

---

## 5. Reference TUI Implementation (`@sullux/tui`)

Channel includes a production-grade terminal user interface built with `@sullux/tui`:
* **Zero-Markup Controller Architecture:** JavaScript controllers return pure domain data and semantic classes (`userCard`, `streamCard`, `planItem`). Presentation, borders, layouts, and colors are defined entirely in declarative `view.yaml` and `theme.yaml`.
* **Abstract Markdown Control:** Renders formatted Markdown via `@sullux/markdown-compiler`, supporting syntax-highlighted code blocks, tables, callouts, and indented hierarchical lists.
* **Responsive 3-Panel Layout:**
  * **Tier 1 ($\ge 160$ cols):** Full 3-pane layout (Chat 60+ cols, Stream 60+ cols, Plan 40+ cols).
  * **Tier 2 ($120 \dots 159$ cols):** Dual-pane layout (Chat + Stream visible; Plan surfaced via hotkey `d`).
  * **Tier 3 ($< 120$ cols):** Single-pane layout (Chat active by default; Stream on `s`, Plan on `d`).
