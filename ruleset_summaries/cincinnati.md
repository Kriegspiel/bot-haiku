# Cincinnati

Play ordinary chess on the true board with opponent pieces and moves hidden. A player may try moves until one legal move stands. Illegal and Nonsense responses are public referee information; the failed try is retracted, and the actual failed move is visible only when the prompt includes it from your own scorecard perspective.

Before a turn, the referee may announce that the mover has a pawn capture. This is binary information: at least one legal pawn capture exists, but the count, source squares, target squares, and captured identities are not given. The announcement does not force a pawn capture, and there is no `ask_any` action. A pawn-capture try without an announcement can be Nonsense.

Legal captures announce the capture square and whether the captured unit was a pawn or a non-pawn piece, so use `material.pawns_captured` when it appears. Checks announce direction only. Promotion and castling are legal but not announced, and promoted piece type stays hidden. Stalemate is an ordinary draw announcement, not RAND's stalemate-loss rule.
