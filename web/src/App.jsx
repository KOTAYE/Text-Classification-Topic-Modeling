import { useEffect, useRef, useState } from 'react'

const EXAMPLES = [
  'Apple reports record quarterly revenue driven by iPhone sales',
  'Manchester United signs new striker ahead of the season',
  'Динамо переграло Шахтар у фіналі кубка',
  'Who is the student behind this project?',
  'Classify this and send it to my Telegram: UN Security Council meets over border conflict',
]

const TOOL_LABELS = {
  classify_news: 'classifier',
  about_student: 'notes',
  send_telegram_message: 'telegram',
  get_telegram_bot_name: 'telegram check',
}

/**
 * The classifier's full opinion, not just its verdict.
 *
 * One hue rather than four: every bar measures the same thing, so colour would
 * only be repeating the label. The winner carries full strength and the rest
 * recede, which puts the eye on the gap — the interesting part when a story
 * splits 82/18 between two topics.
 */
function Distribution({ scores }) {
  const top = scores[0]

  return (
    <div className="dist">
      <div className="dist-head">Classifier confidence</div>
      {scores.map((s) => (
        <div className="row" key={s.label}>
          <span className="row-label">{s.label}</span>
          <div className="track">
            <div
              className={s.label === top.label ? 'fill fill-top' : 'fill'}
              style={{ width: `${Math.max(s.score * 100, 0.6)}%` }}
            />
          </div>
          <span className="row-value">{(s.score * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

/**
 * Proof of delivery, not a claim of it.
 *
 * The receipt is Telegram's own reply — chat and message number — so what the
 * page shows is what the API confirmed, not what the agent said it did.
 */
function Delivered({ delivery }) {
  return (
    <div className="delivery">
      <div className="delivery-head">Sent to Telegram</div>
      <div className="delivery-text">{delivery.text}</div>
      {delivery.receipt && <div className="delivery-receipt">{delivery.receipt}</div>}
    </div>
  )
}

function Message({ turn }) {
  if (turn.role === 'user') {
    return (
      <div className="turn turn-user">
        <div className="bubble">{turn.content}</div>
      </div>
    )
  }

  return (
    <div className="turn turn-agent">
      <div className="bubble">
        {turn.content}
        {turn.tools?.length > 0 && (
          <div className="tools">
            {turn.tools.map((name, i) => (
              <span className="tool" key={`${name}-${i}`}>
                {TOOL_LABELS[name] ?? name}
              </span>
            ))}
          </div>
        )}
      </div>
      {turn.distribution && <Distribution scores={turn.distribution} />}
      {turn.telegram && <Delivered delivery={turn.telegram} />}
    </div>
  )
}

export default function App() {
  const [turns, setTurns] = useState([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [tools, setTools] = useState([])
  const endRef = useRef(null)

  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then((d) => setTools(d.tools ?? []))
      .catch(() => setTools([]))
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns, busy])

  async function send(text) {
    const message = text.trim()
    if (!message || busy) return

    setDraft('')
    setBusy(true)
    const asked = [...turns, { role: 'user', content: message }]
    setTurns(asked)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          history: turns.map(({ role, content }) => ({ role, content })),
        }),
      })
      if (!response.ok) throw new Error(`server returned ${response.status}`)
      const data = await response.json()

      setTurns([
        ...asked,
        {
          role: 'assistant',
          content: data.reply,
          tools: data.tools_used,
          distribution: data.distribution,
          telegram: data.telegram,
        },
      ])
    } catch (err) {
      setTurns([
        ...asked,
        { role: 'assistant', content: `Could not reach the agent: ${err.message}`, error: true },
      ])
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header>
        <div>
          <h1>News topic agent</h1>
          <p className="sub">
            xlm-roberta fine-tuned on AG News, wrapped in an agent with retrieval and Telegram
          </p>
        </div>
        <div className="tool-list">
          {tools.map((name) => (
            <span className="tool" key={name}>
              {TOOL_LABELS[name] ?? name}
            </span>
          ))}
        </div>
      </header>

      <main>
        {turns.length === 0 && (
          <div className="empty">
            <p>Try one of these:</p>
            <div className="examples">
              {EXAMPLES.map((example) => (
                <button key={example} onClick={() => send(example)}>
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((turn, i) => (
          <Message turn={turn} key={i} />
        ))}

        {busy && (
          <div className="turn turn-agent">
            <div className="bubble thinking">thinking…</div>
          </div>
        )}

        <div ref={endRef} />
      </main>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          send(draft)
        }}
      >
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Paste a headline, or ask about the student"
          disabled={busy}
          autoFocus
        />
        <button type="submit" disabled={busy || !draft.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}
