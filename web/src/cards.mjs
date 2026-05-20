// Card encoding matches src/gpu_poker/kernels/card_utils.py:
//   card_idx in [0, 51]
//   rank = card_idx % 13   (0='2', ..., 8='T', 9='J', 10='Q', 11='K', 12='A')
//   suit = card_idx // 13  (0=c, 1=d, 2=h, 3=s)

const RANK_CHARS = "23456789TJQKA";
const SUIT_CHARS = "cdhs";

export const INVALID_CARD = -1;
export const NUM_CARDS = 52;

export function cardRank(idx) {
  if (idx < 0) return -1;
  return idx % 13;
}

export function cardSuit(idx) {
  if (idx < 0) return -1;
  return Math.floor(idx / 13);
}

/** Convert internal card index to pokersolver string ("Ah", "Tc", ...). */
export function cardToString(idx) {
  if (idx < 0) return "??";
  return RANK_CHARS[cardRank(idx)] + SUIT_CHARS[cardSuit(idx)];
}

/** Parse "Ah" -> 12 + 2*13 = 38. */
export function stringToCard(str) {
  const r = RANK_CHARS.indexOf(str[0].toUpperCase());
  const s = SUIT_CHARS.indexOf(str[1].toLowerCase());
  if (r < 0 || s < 0) throw new Error(`bad card string: ${str}`);
  return r + s * 13;
}

/** Shuffled deck [0..51] using the given RNG. */
export function shuffledDeck(rng = Math.random) {
  const deck = Array.from({ length: NUM_CARDS }, (_, i) => i);
  for (let i = deck.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [deck[i], deck[j]] = [deck[j], deck[i]];
  }
  return deck;
}
