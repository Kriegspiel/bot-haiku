# Wild 16

Play ordinary chess on the true board with opponent pieces and moves hidden. Wild 16 keeps illegal tries private to the player who made them; do not infer opponent illegal attempts from silence or missing public log entries. The mover keeps trying until a legal move stands.

Before each turn, the referee publicly announces the exact count of legal pawn captures available to the player to move. The count gives no source squares, target squares, captured identities, or en passant status, and it does not force a pawn capture. There is no `ask_any` action.

Legal captures announce the captured square and whether the captured unit was a pawn or a non-pawn piece; en passant uses the captured pawn's square. Use `material.pawns_captured` when it appears. Checks announce direction only. Promotion and castling are legal but not announced, and promoted piece type stays hidden. Stalemate is an ordinary draw announcement, not RAND's stalemate-loss rule.
