import React, { useState, useRef, useCallback, useEffect, useMemo } from "react"
import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib"

// ── Layout constants ──────────────────────────────────────────────────────────
const ROW_H    = 34
const HEADER_H = 40
const CNT_W    = 30    // stint-count column (far left, holds +/n/− control)
const NAME_W   = 130
const NET_W    = 44    // net RAPM column
const ADD_W    = 24    // "+" button column
const EDGE_HIT = 10
const TOTAL_MINS = 40
const SNAP = 0.5
const DEFAULT_NEW_STINT = 5  // minutes for a freshly added stint

// ── Types ─────────────────────────────────────────────────────────────────────
interface Player {
  player_id: number
  player_name: string
  default_minutes: number
  orapm: number
  drapm: number
}

interface Stint {
  start: number
  end: number
}

type DragType = "move" | "left" | "right"

interface DragState {
  player_id: number
  stint_idx: number
  type: DragType
  mouseX0: number
  stint0: Stint
}

// ── Helpers ───────────────────────────────────────────────────────────────────
const snapV  = (v: number) => Math.round(v / SNAP) * SNAP
const clampV = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

function defaultStints(player: Player): Stint[] {
  const mins = clampV(player.default_minutes, 0, TOTAL_MINS)
  if (mins <= 0) return []
  return [{ start: 0, end: snapV(mins) }]
}

function overlaps(a: Stint, b: Stint): boolean {
  return a.start < b.end - 0.01 && b.start < a.end - 0.01
}

/** Find the largest available gap and place a new stint there (up to DEFAULT_NEW_STINT wide). */
function firstGap(stints: Stint[]): Stint | null {
  const sorted = [...stints].sort((a, b) => a.start - b.start)
  let cursor = 0
  let best: { start: number; size: number } | null = null

  for (const s of sorted) {
    const size = snapV(s.start - cursor)
    if (size >= SNAP && (best === null || size > best.size)) {
      best = { start: cursor, size }
    }
    cursor = s.end
  }
  const trailing = snapV(TOTAL_MINS - cursor)
  if (trailing >= SNAP && (best === null || trailing > best.size)) {
    best = { start: cursor, size: trailing }
  }

  if (!best) return null
  const dur = Math.min(DEFAULT_NEW_STINT, best.size)
  return { start: snapV(best.start), end: snapV(best.start + dur) }
}

function totalMins(stints: Stint[]): number {
  return stints.reduce((sum, s) => sum + Math.max(0, snapV(s.end - s.start)), 0)
}

// ── Component ─────────────────────────────────────────────────────────────────
function RotationChart({ args }: ComponentProps) {
  const players: Player[]     = args.players     ?? []
  const label: string         = args.label       ?? ""
  const teamColor: string     = args.team_color  ?? "#1e3a5f"
  const textColor: string     = args.text_color  ?? "#FFFFFF"
  const teamKey: string       = args.team_key    ?? "default"
  const forcedZeros: number[] = args.forced_zeros ?? []

  // State: array of stints per player
  const [stints, setStints] = useState<Record<number, Stint[]>>(() =>
    Object.fromEntries(players.map((p) => [p.player_id, defaultStints(p)]))
  )

  // Reset when team changes
  const prevKeyRef = useRef(teamKey)
  useEffect(() => {
    if (prevKeyRef.current !== teamKey) {
      prevKeyRef.current = teamKey
      setStints(Object.fromEntries(players.map((p) => [p.player_id, defaultStints(p)])))
    }
  }, [teamKey, players])

  // Zero stints for DNP players; restore to default when un-DNP'd
  const prevForcedRef = useRef<number[]>([])
  useEffect(() => {
    const prev = prevForcedRef.current
    const toZero    = forcedZeros.filter(pid => !prev.includes(pid))
    const toRestore = prev.filter(pid => !forcedZeros.includes(pid))
    if (toZero.length === 0 && toRestore.length === 0) return
    setStints(cur => {
      const next = { ...cur }
      for (const pid of toZero) {
        next[pid] = []
      }
      for (const pid of toRestore) {
        const p = players.find(pl => pl.player_id === pid)
        if (p) next[pid] = defaultStints(p)
      }
      return next
    })
    prevForcedRef.current = forcedZeros
  }, [forcedZeros, players])

  const dragRef = useRef<DragState | null>(null)
  const svgRef  = useRef<SVGSVGElement>(null)
  const [svgWidth, setSvgWidth] = useState(560)

  const timeW = svgWidth - CNT_W - NAME_W - NET_W - ADD_W

  const toX    = useCallback((m: number) => CNT_W + NAME_W + (m / TOTAL_MINS) * timeW, [timeW])
  const toMins = useCallback((x: number) => ((x - CNT_W - NAME_W) / timeW) * TOTAL_MINS, [timeW])

  // Emit total minutes per player whenever stints change
  useEffect(() => {
    const out: Record<number, number> = {}
    players.forEach((p) => {
      out[p.player_id] = totalMins(stints[p.player_id] ?? [])
    })
    Streamlit.setComponentValue(out)
  }, [stints, players])

  useEffect(() => {
    Streamlit.setFrameHeight(HEADER_H + players.length * ROW_H + 6)
  }, [players.length])

  useEffect(() => {
    const el = svgRef.current?.parentElement
    if (!el) return
    const ro = new ResizeObserver((e) => setSvgWidth(e[0].contentRect.width || 560))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // ── Drag handlers ──
  const onMouseMove = useCallback((e: MouseEvent) => {
    const drag = dragRef.current
    if (!drag || !svgRef.current) return
    const dMins = ((e.clientX - drag.mouseX0) / timeW) * TOTAL_MINS
    const s0 = drag.stint0
    const dur = s0.end - s0.start

    setStints((prev) => {
      const arr = [...(prev[drag.player_id] ?? [])]
      const others = arr.filter((_, i) => i !== drag.stint_idx)

      let ns: Stint
      if (drag.type === "move") {
        const ns_start = clampV(snapV(s0.start + dMins), 0, TOTAL_MINS - dur)
        ns = { start: ns_start, end: snapV(ns_start + dur) }
      } else if (drag.type === "left") {
        ns = { start: clampV(snapV(s0.start + dMins), 0, s0.end - SNAP), end: s0.end }
      } else {
        ns = { start: s0.start, end: clampV(snapV(s0.end + dMins), s0.start + SNAP, TOTAL_MINS) }
      }

      // Don't allow overlapping other stints of same player
      if (others.some((o) => overlaps(ns, o))) return prev

      arr[drag.stint_idx] = ns
      return { ...prev, [drag.player_id]: arr }
    })
  }, [timeW])

  const onMouseUp = useCallback(() => { dragRef.current = null }, [])

  useEffect(() => {
    window.addEventListener("mousemove", onMouseMove)
    window.addEventListener("mouseup", onMouseUp)
    return () => {
      window.removeEventListener("mousemove", onMouseMove)
      window.removeEventListener("mouseup", onMouseUp)
    }
  }, [onMouseMove, onMouseUp])

  const startDrag = (e: React.MouseEvent, pid: number, idx: number, type: DragType) => {
    e.preventDefault()
    e.stopPropagation()
    const s = (stints[pid] ?? [])[idx]
    if (!s) return
    dragRef.current = { player_id: pid, stint_idx: idx, type, mouseX0: e.clientX, stint0: { ...s } }
  }

  const removeStint = (e: React.MouseEvent, pid: number, idx: number) => {
    e.preventDefault()
    setStints((prev) => ({ ...prev, [pid]: (prev[pid] ?? []).filter((_, i) => i !== idx) }))
  }

  const addStint = (pid: number) => {
    setStints((prev) => {
      const gap = firstGap(prev[pid] ?? [])
      if (!gap) return prev
      return { ...prev, [pid]: [...(prev[pid] ?? []), gap] }
    })
  }

  const removeLastStint = (pid: number) => {
    setStints((prev) => {
      const arr = prev[pid] ?? []
      if (arr.length === 0) return prev
      return { ...prev, [pid]: arr.slice(0, -1) }
    })
  }

  // Activate a benched player by pulling from left edge
  const activatePlayer = (e: React.MouseEvent, pid: number) => {
    const newStint: Stint = { start: 0, end: SNAP }
    setStints((prev) => ({ ...prev, [pid]: [newStint] }))
    dragRef.current = { player_id: pid, stint_idx: 0, type: "right", mouseX0: e.clientX, stint0: { ...newStint } }
  }

  const POSITIONS = ["PG", "SG", "SF", "PF", "C"]

  // Sort players by current total minutes descending — drives row order + position labels
  const sortedPlayers = useMemo(() => {
    return [...players].sort((a, b) =>
      totalMins(stints[b.player_id] ?? []) - totalMins(stints[a.player_id] ?? [])
    )
  }, [players, stints])

  // Assign PG/SG/SF/PF/C to top-5 players with minutes > 0
  const posMap = useMemo(() => {
    const m: Record<number, string> = {}
    let slot = 0
    for (const p of sortedPlayers) {
      if (slot >= 5) break
      if (totalMins(stints[p.player_id] ?? []) > 0) {
        m[p.player_id] = POSITIONS[slot++]
      }
    }
    return m
  }, [sortedPlayers, stints])

  const quarters = [0, 10, 20, 30, 40]
  const totalH   = HEADER_H + players.length * ROW_H

  return (
    <div style={{ width: "100%", userSelect: "none" }}>
      {label && (
        <div style={{ fontSize: 13, fontWeight: 700, color: "#333", marginBottom: 4, paddingLeft: NAME_W }}>
          {label}
        </div>
      )}
      <svg ref={svgRef} width="100%" height={totalH}
        style={{ display: "block", fontFamily: "system-ui, sans-serif" }}>

        {/* ── Header ── */}
        {quarters.map((q, i) => (
          <line key={q} x1={toX(q)} y1={HEADER_H - 6} x2={toX(q)} y2={totalH}
            stroke={i === 0 || i === 4 ? "#bbb" : "#e0e0e0"}
            strokeWidth={i === 0 || i === 4 ? 1 : 0.8} />
        ))}
        {[0, 1, 2, 3].map((q) => (
          <text key={q} x={toX(q * 10 + 5)} y={HEADER_H - 10}
            textAnchor="middle" fontSize={10} fill="#999">Q{q + 1}</text>
        ))}
        {/* Count column header */}
        <text x={CNT_W / 2} y={HEADER_H - 10} textAnchor="middle" fontSize={9} fill="#aaa">stints</text>
        <text x={CNT_W + NAME_W - 8} y={HEADER_H - 10} textAnchor="end" fontSize={10} fill="#999">Player</text>
        <text x={svgWidth - NET_W - ADD_W + 4} y={HEADER_H - 10} fontSize={10} fill="#999">net</text>
        <text x={svgWidth - ADD_W / 2} y={HEADER_H - 10} textAnchor="middle" fontSize={11} fill="#bbb">+</text>

        {/* ── Rows ── */}
        {sortedPlayers.map((player, i) => {
          const y          = HEADER_H + i * ROW_H
          const ps         = stints[player.player_id] ?? []
          const tot        = totalMins(ps)
          const net        = player.orapm + player.drapm
          const netColor   = net >= 0 ? "#2a9d8f" : "#e76f51"
          const shortName  = player.player_name.split(" ").slice(-1)[0]
          const canAdd     = firstGap(ps) !== null
          const posLabel   = posMap[player.player_id] ?? ""

          return (
            <g key={player.player_id}>
              {/* Row background */}
              <rect x={0} y={y} width={svgWidth} height={ROW_H}
                fill={i % 2 === 0 ? "#fafafa" : "#f4f4f4"} />

              {/* Quarter grid lines */}
              {quarters.map((q) => (
                <line key={q} x1={toX(q)} y1={y} x2={toX(q)} y2={y + ROW_H}
                  stroke="#e8e8e8" strokeWidth={0.5} />
              ))}

              {/* ── Stint count control: + / n / − ── */}
              {/* + button (top) */}
              <text x={CNT_W / 2} y={y + 11}
                textAnchor="middle" fontSize={13} fontWeight={400}
                fill={canAdd ? teamColor : "#ccc"}
                style={{ cursor: canAdd ? "pointer" : "default" }}
                onClick={(e) => { e.stopPropagation(); canAdd && addStint(player.player_id) }}>
                +
              </text>
              {/* count (middle) */}
              <text x={CNT_W / 2} y={y + ROW_H / 2 + 4}
                textAnchor="middle" fontSize={11} fontWeight={700}
                fill={ps.length > 0 ? teamColor : "#bbb"}>
                {ps.length}
              </text>
              {/* − button (bottom) */}
              <text x={CNT_W / 2} y={y + ROW_H - 4}
                textAnchor="middle" fontSize={14} fontWeight={400}
                fill={ps.length > 0 ? "#e76f51" : "#ccc"}
                style={{ cursor: ps.length > 0 ? "pointer" : "default" }}
                onClick={(e) => { e.stopPropagation(); ps.length > 0 && removeLastStint(player.player_id) }}>
                −
              </text>

              {/* Position label (left of name, reactive) */}
              {posLabel && (
                <text x={CNT_W + 4} y={y + ROW_H / 2 + 4}
                  fontSize={8} fontWeight={700} fill={teamColor} opacity={0.7}>
                  {posLabel}
                </text>
              )}
              {/* Player name */}
              <text x={CNT_W + NAME_W - 8} y={y + ROW_H / 2 + 4}
                textAnchor="end" fontSize={11} fill={tot > 0 ? "#333" : "#aaa"}>
                <title>{player.player_name}</title>
                {shortName}
              </text>

              {/* Stint blocks */}
              {ps.map((s, si) => {
                const x1     = toX(s.start)
                const x2     = toX(s.end)
                const blockW = Math.max(0, x2 - x1)
                const mins   = snapV(s.end - s.start)
                return (
                  <g key={si}
                    onMouseDown={(e) => startDrag(e, player.player_id, si, "move")}
                    onDoubleClick={(e) => removeStint(e, player.player_id, si)}
                    style={{ cursor: "grab" }}>
                    <rect x={x1} y={y + 5} width={blockW} height={ROW_H - 10}
                      rx={3} fill={teamColor} opacity={0.88} />
                    {blockW > 18 && (
                      <text x={(x1 + x2) / 2} y={y + ROW_H / 2 + 4}
                        textAnchor="middle" fontSize={10} fill={textColor} fontWeight={700}
                        style={{ pointerEvents: "none" }}>
                        {mins % 1 === 0 ? mins.toFixed(0) : mins.toFixed(1)}
                      </text>
                    )}
                    {/* Left resize */}
                    <rect x={x1} y={y + 5} width={EDGE_HIT} height={ROW_H - 10}
                      fill="transparent" style={{ cursor: "ew-resize" }}
                      onMouseDown={(e) => startDrag(e, player.player_id, si, "left")} />
                    {/* Right resize */}
                    <rect x={x2 - EDGE_HIT} y={y + 5} width={EDGE_HIT} height={ROW_H - 10}
                      fill="transparent" style={{ cursor: "ew-resize" }}
                      onMouseDown={(e) => startDrag(e, player.player_id, si, "right")} />
                  </g>
                )
              })}

              {/* Pull-handle for benched players */}
              {ps.length === 0 && (
                <rect x={toX(0)} y={y + 5} width={6} height={ROW_H - 10}
                  rx={2} fill="#ccc" opacity={0.5} style={{ cursor: "ew-resize" }}
                  onMouseDown={(e) => activatePlayer(e, player.player_id)} />
              )}

              {/* Net RAPM */}
              <text x={svgWidth - NET_W - ADD_W + 4} y={y + ROW_H / 2 + 4}
                fontSize={10} fill={netColor} textAnchor="start">
                {net >= 0 ? "+" : ""}{net.toFixed(1)}
              </text>

              {/* Total minutes sub-label (shown when player has multiple stints) */}
              {ps.length > 1 && (
                <text x={CNT_W + NAME_W - 8} y={y + ROW_H - 3}
                  textAnchor="end" fontSize={8} fill="#888">
                  {tot.toFixed(1)}m
                </text>
              )}

              {/* Add stint "+" button */}
              <text x={svgWidth - ADD_W / 2} y={y + ROW_H / 2 + 5}
                textAnchor="middle" fontSize={16} fontWeight={300}
                fill={canAdd ? teamColor : "#ddd"}
                style={{ cursor: canAdd ? "pointer" : "default" }}
                onClick={() => canAdd && addStint(player.player_id)}>
                +
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

export default withStreamlitConnection(RotationChart)
