# Berkeley + Any

Play ordinary chess on the true board with opponent pieces and moves hidden. You see your private FEN, public referee log, public material fields, possible actions, and the exact legal moves supplied in the turn payload. A turn continues until a legal move stands. Illegal tries receive a public rejection, but the tried move itself is not revealed to the opponent unless it is shown from that player's own scorecard perspective.

Legal captures announce only the capture square. Captured identity, capturing piece, promotion, castling, and en passant identity are not announced; en passant is treated as an ordinary capture. Checks announce direction only: rank, file, long diagonal, short diagonal, knight, or multiple directions when applicable.

`ask_any` asks whether the side to move has at least one legal pawn capture. A positive answer means this turn must finish with a legal pawn capture. There is no pawn-capture count and no captured-pawn identity, so `pawns_captured` is absent. Stalemate is an ordinary draw announcement, not RAND's stalemate-loss rule.
