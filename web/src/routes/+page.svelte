<script>
  import { onMount } from "svelte";
  import {
    newHand,
    applyAction,
    legalActions,
    isHandComplete,
    stateForEncoder,
    nextButton,
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_RAISE,
  } from "../engine.mjs";
  import { cardToString } from "../cards.mjs";

  let bot = $state(null);
  let modelLoaded = $state(false);
  let modelLoading = $state(false);
  let loadError = $state("");
  let humanSeat = $state(0); // human is player 0, bot is player 1
  let button = $state(0);
  // Defaults match the training stake (50/100 blinds, 100 BB starting stacks
  // sit in the middle of the 40-200 BB randomized training distribution).
  let humanStack = $state(10000);
  let botStack = $state(10000);
  let smallBlind = $state(50);
  let bigBlind = $state(100);
  let state = $state(null);
  let history = $state([]); // per-hand action log
  let botThinking = $state(false);

  async function loadModelFromFile(file) {
    modelLoading = true;
    loadError = "";
    try {
      // Dynamic import so ORT-Web isn't pulled into SSR / initial bundle.
      const { PokerBot } = await import("../inference.mjs");
      bot = new PokerBot();
      await bot.loadModel(file);
      modelLoaded = true;
    } catch (e) {
      loadError = e.message;
      console.error(e);
    } finally {
      modelLoading = false;
    }
  }

  function onDrop(e) {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) loadModelFromFile(f);
  }
  function onPick(e) {
    const f = e.target.files[0];
    if (f) loadModelFromFile(f);
  }

  function startHand() {
    // Re-buy when either stack is too low to even post the BB. Pick a starting
    // stack that sits inside the 40-200 BB training distribution (100 BB).
    const defaultStack = 100 * bigBlind;
    if (humanStack < bigBlind || botStack < bigBlind) {
      humanStack = defaultStack;
      botStack = defaultStack;
    }
    const stacks = humanSeat === 0 ? [humanStack, botStack] : [botStack, humanStack];
    // `startingStack` is the per-env normalization constant the obs encoder
    // uses; it should match the actual starting stack at hand start so chip
    // ratios land in the training distribution.
    state = newHand({
      button,
      stacks,
      smallBlind,
      bigBlind,
      startingStack: Math.max(humanStack, botStack),
    });
    history = [];
    if (bot) bot.resetHidden();
    maybeBotMove();
  }

  function endHandAndTally() {
    // After hand ends, sync the local stack vars from the state and rotate button.
    humanStack = state.stacks[humanSeat];
    botStack = state.stacks[1 - humanSeat];
    button = nextButton(button);
  }

  function actionName(a) {
    return ["FOLD", "CHECK", "CALL", "RAISE"][a];
  }

  async function maybeBotMove() {
    while (state && !isHandComplete(state) && state.active_player !== humanSeat) {
      botThinking = true;
      // Yield to render the spinner.
      await new Promise((r) => setTimeout(r, 30));
      const decision = await bot.act(stateForEncoder(state));
      let amount = 0;
      if (decision.actionType === ACTION_RAISE) {
        const la = legalActions(state);
        const { bucketToAmount } = await import("../inference.mjs");
        const potTotal = state.pot + state.bets[0] + state.bets[1];
        amount = bucketToAmount({
          bucket: decision.raiseBucket,
          potTotal,
          minRaise: la.min_raise,
          maxRaise: la.max_raise,
        });
      }
      const note = describeAction(state, decision.actionType, amount, "bot");
      history = [...history, note];
      applyAction(state, decision.actionType, amount);
      botThinking = false;
      if (isHandComplete(state)) endHandAndTally();
    }
  }

  function humanAct(action, amount = 0) {
    if (!state || isHandComplete(state) || state.active_player !== humanSeat) return;
    const note = describeAction(state, action, amount, "you");
    history = [...history, note];
    applyAction(state, action, amount);
    if (isHandComplete(state)) {
      endHandAndTally();
    } else {
      maybeBotMove();
    }
  }

  function describeAction(s, action, amount, who) {
    const stageName = ["preflop", "flop", "turn", "river", "showdown", "terminal"][s.stage];
    if (action === ACTION_FOLD) return `${who} folds (${stageName})`;
    if (action === ACTION_CHECK) return `${who} checks (${stageName})`;
    if (action === ACTION_CALL) return `${who} calls (${stageName})`;
    return `${who} raises ${amount} (${stageName})`;
  }

  let humanCards = $derived(state ? state.hole_cards[humanSeat] : []);
  let botCards = $derived(
    state && isHandComplete(state) && state.winner !== -1 ? state.hole_cards[1 - humanSeat] : [-1, -1],
  );
  let board = $derived(state ? stateForEncoder(state).community_cards : [-1, -1, -1, -1, -1]);
  let myLegal = $derived(state && !isHandComplete(state) && state.active_player === humanSeat ? legalActions(state) : null);
  let toCall = $derived(state && !isHandComplete(state) ? Math.max(0, state.bets[1 - state.active_player] - state.bets[state.active_player]) : 0);

  let showHints = $state(false);
  onMount(() => {
    if (typeof localStorage !== "undefined") {
      showHints = localStorage.getItem("pokergpu.showHints") === "1";
    }
  });
  function toggleHints() {
    showHints = !showHints;
    if (typeof localStorage !== "undefined") {
      localStorage.setItem("pokergpu.showHints", showHints ? "1" : "0");
    }
  }

  let raiseSlider = $state(0);
  $effect(() => {
    if (myLegal) raiseSlider = myLegal.min_raise;
  });

  function clampRaise(v) {
    if (!myLegal) return 0;
    return Math.max(myLegal.min_raise, Math.min(myLegal.max_raise, Math.round(v)));
  }

  function onKeyDown(e) {
    // Ignore when typing into a real input (we don't have any text inputs in
    // the play view, but the slider also lives in an <input>; let its native
    // arrow-key handling pass through).
    const t = e.target;
    if (t && t.tagName === "INPUT" && t.type !== "range") return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (!modelLoaded) return;

    if (e.key === "?") { e.preventDefault(); toggleHints(); return; }

    if (e.key === "Enter") {
      if (!state || isHandComplete(state)) {
        e.preventDefault();
        startHand();
      }
      return;
    }
    if (!myLegal || botThinking) return;

    const k = e.key.toLowerCase();
    if (k === "f" && myLegal.action_mask[ACTION_FOLD]) { e.preventDefault(); humanAct(ACTION_FOLD); return; }
    if (k === "c") {
      if (myLegal.action_mask[ACTION_CHECK]) { e.preventDefault(); humanAct(ACTION_CHECK); return; }
      if (myLegal.action_mask[ACTION_CALL])  { e.preventDefault(); humanAct(ACTION_CALL);  return; }
      return;
    }
    if (k === "r" && myLegal.action_mask[ACTION_RAISE]) { e.preventDefault(); humanAct(ACTION_RAISE, raiseSlider); return; }

    if (!myLegal.action_mask[ACTION_RAISE]) return;
    const potTotal = state.pot + state.bets[0] + state.bets[1];
    if (k === "1") { e.preventDefault(); raiseSlider = clampRaise(potTotal / 3);   return; }
    if (k === "2") { e.preventDefault(); raiseSlider = clampRaise(potTotal / 2);   return; }
    if (k === "3") { e.preventDefault(); raiseSlider = clampRaise(potTotal * 2/3); return; }
    if (k === "4") { e.preventDefault(); raiseSlider = clampRaise(potTotal);       return; }
    if (k === "5") { e.preventDefault(); raiseSlider = myLegal.max_raise;          return; }
    if (k === "m") { e.preventDefault(); raiseSlider = myLegal.min_raise;          return; }
    if (e.key === "ArrowUp")   { e.preventDefault(); raiseSlider = clampRaise(raiseSlider + bigBlind); return; }
    if (e.key === "ArrowDown") { e.preventDefault(); raiseSlider = clampRaise(raiseSlider - bigBlind); return; }
  }

  const RANK_LABELS = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"];
  const SUIT_GLYPHS = ["♣","♦","♥","♠"]; // c d h s

  function cardRankLabel(c) {
    if (c < 0) return "";
    return RANK_LABELS[c % 13];
  }
  function cardSuitGlyph(c) {
    if (c < 0) return "";
    return SUIT_GLYPHS[Math.floor(c / 13)];
  }
  function cardIsRed(c) {
    if (c < 0) return false;
    const s = Math.floor(c / 13);
    return s === 1 || s === 2;
  }
  // Slot kind: "empty" = no card yet (—), "back" = face-down, "face" = real card.
  function cardKind(c, faceDown) {
    if (c < 0) return faceDown ? "back" : "empty";
    return "face";
  }
</script>

<svelte:head>
  <title>pokergpu — play vs your trained bot</title>
</svelte:head>

<svelte:window onkeydown={onKeyDown} />

<main>
  <header>
    <h1>pokergpu</h1>
    <p class="tag">Drop a <code>.onnx</code> file you exported and play heads-up.</p>
  </header>

  {#if !modelLoaded}
    <section
      class="drop"
      role="region"
      aria-label="Drop ONNX model here"
      ondragover={(e) => e.preventDefault()}
      ondrop={onDrop}
    >
      <div class="drop-icon" aria-hidden="true">↓</div>
      <p class="drop-title">{modelLoading ? "loading model…" : "Drop your .onnx model here"}</p>
      <p class="drop-sub">or pick a file from disk</p>
      <label class="file-btn">
        <span>Choose file</span>
        <input type="file" accept=".onnx" onchange={onPick} disabled={modelLoading} />
      </label>
      {#if loadError}<p class="err">{loadError}</p>{/if}
      <p class="drop-hint">
        Generate one with
        <code>uv run python train/export_onnx.py --checkpoint X.pt --out model.onnx</code>
      </p>
    </section>
  {:else}
    <section class="game">
      <div class="config">
        <span>Stacks: {humanStack} vs {botStack}</span>
        <span>Blinds: {smallBlind}/{bigBlind}</span>
        <span>Button: {button === humanSeat ? "you" : "bot"}</span>
        <button
          class="hint-toggle"
          class:active={showHints}
          title="Toggle keyboard shortcut hints (?)"
          aria-label="Toggle keyboard shortcut hints"
          aria-pressed={showHints}
          onclick={toggleHints}
        >⌨</button>
        <button onclick={startHand}>{state ? "Deal next hand" : "Deal first hand"}{#if showHints} <kbd>⏎</kbd>{/if}</button>
      </div>

      {#if state}
        <div class="board">
          <div class="row">
            <span class="label">Bot:</span>
            <div class="cards">
              {#each botCards as c}
                {@const kind = cardKind(c, !isHandComplete(state))}
                <span class="card card-{kind}" class:red={cardIsRed(c)}>
                  {#if kind === "face"}
                    <span class="rank">{cardRankLabel(c)}</span>
                    <span class="suit">{cardSuitGlyph(c)}</span>
                    <span class="rank rank-br">{cardRankLabel(c)}</span>
                  {/if}
                </span>
              {/each}
            </div>
            <span class="stack">stack {state.stacks[1 - humanSeat]}</span>
            <span class="bet">bet {state.bets[1 - humanSeat]}</span>
          </div>
          <div class="row board-row">
            <span class="label">Board:</span>
            <div class="cards">
              {#each board as c}
                {@const kind = cardKind(c, false)}
                <span class="card card-{kind}" class:red={cardIsRed(c)}>
                  {#if kind === "face"}
                    <span class="rank">{cardRankLabel(c)}</span>
                    <span class="suit">{cardSuitGlyph(c)}</span>
                    <span class="rank rank-br">{cardRankLabel(c)}</span>
                  {/if}
                </span>
              {/each}
            </div>
            <span class="pot">pot {state.pot + state.bets[0] + state.bets[1]}</span>
          </div>
          <div class="row">
            <span class="label">You:</span>
            <div class="cards">
              {#each humanCards as c}
                {@const kind = cardKind(c, false)}
                <span class="card card-{kind}" class:red={cardIsRed(c)}>
                  {#if kind === "face"}
                    <span class="rank">{cardRankLabel(c)}</span>
                    <span class="suit">{cardSuitGlyph(c)}</span>
                    <span class="rank rank-br">{cardRankLabel(c)}</span>
                  {/if}
                </span>
              {/each}
            </div>
            <span class="stack">stack {state.stacks[humanSeat]}</span>
            <span class="bet">bet {state.bets[humanSeat]}</span>
          </div>
        </div>

        {#if !isHandComplete(state)}
          <div class="actions">
            {#if botThinking}
              <p class="thinking">bot is thinking…</p>
            {:else if myLegal}
              <button disabled={!myLegal.action_mask[ACTION_FOLD]} onclick={() => humanAct(ACTION_FOLD)}>Fold{#if showHints} <kbd>F</kbd>{/if}</button>
              <button disabled={!myLegal.action_mask[ACTION_CHECK]} onclick={() => humanAct(ACTION_CHECK)}>Check{#if showHints} <kbd>C</kbd>{/if}</button>
              <button disabled={!myLegal.action_mask[ACTION_CALL]} onclick={() => humanAct(ACTION_CALL)}>Call {toCall}{#if showHints} <kbd>C</kbd>{/if}</button>
              <div class="raise">
                <button disabled={!myLegal.action_mask[ACTION_RAISE]} onclick={() => humanAct(ACTION_RAISE, raiseSlider)}>
                  Raise to {state.bets[humanSeat] + toCall + raiseSlider}{#if showHints} <kbd>R</kbd>{/if}
                </button>
                <input
                  type="range"
                  min={myLegal.min_raise}
                  max={Math.max(myLegal.min_raise, myLegal.max_raise)}
                  bind:value={raiseSlider}
                  disabled={!myLegal.action_mask[ACTION_RAISE]}
                />
                <span class="raise-amt">+{raiseSlider}</span>
              </div>
            {:else}
              <p class="thinking">waiting for bot…</p>
            {/if}
            {#if showHints && myLegal && !botThinking && myLegal.action_mask[ACTION_RAISE]}
              <p class="shortcuts">
                <kbd>1</kbd> ⅓ · <kbd>2</kbd> ½ · <kbd>3</kbd> ⅔ · <kbd>4</kbd> pot · <kbd>5</kbd> all-in · <kbd>M</kbd> min · <kbd>↑</kbd><kbd>↓</kbd> ±BB
              </p>
            {/if}
          </div>
        {:else}
          <div class="result">
            {#if state.winner === humanSeat}
              <p class="win">You won this hand (+{state.rewards[humanSeat]})</p>
            {:else if state.winner === 1 - humanSeat}
              <p class="loss">Bot won this hand ({state.rewards[humanSeat]})</p>
            {:else}
              <p>Split pot</p>
            {/if}
          </div>
        {/if}

        {#if history.length > 0}
          <details class="history">
            <summary>history</summary>
            <ol>
              {#each history as note}<li>{note}</li>{/each}
            </ol>
          </details>
        {/if}
      {:else}
        <p class="hint">Hit "Deal first hand" to start.</p>
      {/if}
    </section>
  {/if}
</main>

<style>
  :global(:root) {
    --c-bg: #0f1115;
    --c-panel: #181a20;
    --c-ink: #e6e6e6;
    --c-muted: #6a6f7a;
    --c-red: #e06b6b;
    --c-accent: #6ab0ff;
    --c-good: #6bd97a;
  }
  :global(body) {
    background: var(--c-bg);
    color: var(--c-ink);
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    margin: 0;
  }
  main {
    max-width: 720px;
    margin: 0 auto;
    padding: 32px 16px;
  }
  header h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 600;
    letter-spacing: -0.02em;
  }
  .tag {
    color: var(--c-muted);
    margin: 4px 0 24px;
    font-size: 14px;
  }
  .drop {
    position: relative;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 36px 28px;
    text-align: center;
    background:
      radial-gradient(circle at 50% 0%, rgba(106, 176, 255, 0.08), transparent 60%),
      var(--c-panel);
    transition: border-color 0.15s ease, background 0.15s ease;
  }
  .drop:hover {
    border-color: rgba(106, 176, 255, 0.4);
  }
  .drop-icon {
    width: 44px;
    height: 44px;
    margin: 0 auto 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: rgba(106, 176, 255, 0.12);
    color: var(--c-accent);
    font-size: 22px;
    line-height: 1;
  }
  .drop-title {
    margin: 0 0 4px;
    font-size: 16px;
    font-weight: 600;
    color: var(--c-ink);
    letter-spacing: -0.01em;
  }
  .drop-sub {
    margin: 0 0 18px;
    color: var(--c-muted);
    font-size: 13px;
  }
  .file-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 16px;
    background: var(--c-accent);
    color: #0f1115;
    border: 0;
    border-radius: 6px;
    cursor: pointer;
    font-family: inherit;
    font-size: 13px;
    font-weight: 600;
  }
  .file-btn:hover {
    filter: brightness(1.1);
  }
  .file-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .file-name {
    margin-left: 10px;
    color: var(--c-muted);
    font-size: 13px;
  }
  .drop input[type="file"] {
    display: none;
  }
  .drop-hint {
    margin-top: 22px;
    padding-top: 18px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    color: var(--c-muted);
    font-size: 12px;
  }
  .hint {
    color: var(--c-muted);
    font-size: 12px;
    margin: 8px 0 0;
  }
  .err {
    color: var(--c-red);
    margin: 8px 0 0;
  }
  code {
    background: rgba(255, 255, 255, 0.05);
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 12px;
  }
  .config {
    display: flex;
    flex-wrap: wrap;
    gap: 12px 20px;
    align-items: center;
    background: var(--c-panel);
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 14px;
    margin-bottom: 16px;
  }
  .config > button:not(.hint-toggle) {
    margin-left: auto;
    padding: 8px 14px;
    background: var(--c-accent);
    color: #0f1115;
    border: 0;
    border-radius: 6px;
    cursor: pointer;
    font-family: inherit;
    font-size: 13px;
    font-weight: 600;
  }
  .config > button:not(.hint-toggle):hover {
    filter: brightness(1.1);
  }
  .hint-toggle {
    margin-left: auto;
    width: 30px;
    height: 30px;
    padding: 0;
    background: transparent;
    color: var(--c-muted);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    line-height: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }
  .hint-toggle + button {
    margin-left: 8px !important;
  }
  .hint-toggle:hover {
    color: var(--c-ink);
    border-color: var(--c-accent);
  }
  .hint-toggle.active {
    color: var(--c-accent);
    border-color: var(--c-accent);
    background: rgba(106, 176, 255, 0.08);
  }
  .board {
    background: var(--c-panel);
    border-radius: 8px;
    padding: 16px;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
  }
  .row .label {
    width: 50px;
    color: var(--c-muted);
    font-size: 13px;
  }
  .board-row {
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    margin: 4px 0;
  }
  .cards {
    display: flex;
    gap: 6px;
  }
  .card {
    position: relative;
    width: 42px;
    height: 58px;
    border-radius: 5px;
    background: #fafafa;
    color: #1a1d22;
    box-shadow: 0 1px 0 rgba(0, 0, 0, 0.35), 0 2px 4px rgba(0, 0, 0, 0.35);
    font-family: ui-sans-serif, system-ui, sans-serif;
    user-select: none;
    flex-shrink: 0;
  }
  .card.red {
    color: #c9304a;
  }
  .card .rank {
    position: absolute;
    top: 3px;
    left: 5px;
    font-size: 13px;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.02em;
  }
  .card .rank-br {
    top: auto;
    left: auto;
    bottom: 3px;
    right: 5px;
    transform: rotate(180deg);
  }
  .card .suit {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    line-height: 1;
  }
  .card-empty {
    background: transparent;
    box-shadow: none;
    border: 1px dashed rgba(255, 255, 255, 0.14);
  }
  .card-back {
    background:
      repeating-linear-gradient(45deg, #2a3550 0 4px, #364266 4px 8px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 1px 0 rgba(0, 0, 0, 0.4), 0 2px 4px rgba(0, 0, 0, 0.4);
  }
  .stack,
  .bet,
  .pot {
    margin-left: auto;
    color: var(--c-muted);
    font-size: 13px;
  }
  .bet {
    margin-left: 12px;
  }
  .pot {
    color: var(--c-accent);
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 16px;
  }
  .actions button {
    padding: 10px 16px;
    background: var(--c-panel);
    color: var(--c-ink);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    cursor: pointer;
    font-family: inherit;
    font-size: 14px;
  }
  .actions button:hover:not(:disabled) {
    border-color: var(--c-accent);
  }
  .actions button:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
  .raise {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
    min-width: 220px;
  }
  .raise input[type="range"] {
    flex: 1;
  }
  .raise-amt {
    color: var(--c-muted);
    font-size: 13px;
    width: 60px;
  }
  .thinking {
    color: var(--c-muted);
    font-size: 14px;
  }
  kbd {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 5px;
    border-radius: 3px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: var(--c-muted);
    font-family: inherit;
    font-size: 11px;
    line-height: 1.3;
    vertical-align: 1px;
  }
  .config button kbd {
    background: rgba(0, 0, 0, 0.18);
    border-color: rgba(0, 0, 0, 0.2);
    color: rgba(15, 17, 21, 0.7);
  }
  .shortcuts {
    flex-basis: 100%;
    margin: 4px 0 0;
    color: var(--c-muted);
    font-size: 12px;
  }
  .shortcuts kbd {
    margin: 0 4px 0 0;
  }
  .result {
    text-align: center;
    margin-top: 16px;
  }
  .win {
    color: var(--c-good);
  }
  .loss {
    color: var(--c-red);
  }
  .history {
    margin-top: 16px;
    color: var(--c-muted);
    font-size: 13px;
  }
  .history ol {
    margin: 8px 0 0;
    padding-left: 20px;
  }
</style>
