# RAND

Play ordinary chess on the true board with opponent pieces and moves hidden. RAND makes illegal tries public: normal rebuffs, no answers, and special impossible-move rejections can appear in the referee log. The rejected move itself is only known when the scorecard perspective exposes it.

Before each turn, the referee announces the source squares of the mover's pawns that currently have legal capture tries. This reveals which pawns may capture, but not target squares, captured identities, or en passant status. If the player is in check, only pawn captures that legally answer the check are included. These announcements do not force a pawn capture, and there is no `ask_any` action.

Legal captures announce the capture square and whether the captured unit was a pawn or a non-pawn piece, so use `material.pawns_captured` when it appears. Checks announce direction only. Promotions are announced in RAND, but not the promoted piece type and not the promotion square. The stalemated player loses in RAND; do not treat stalemate as an ordinary draw here.
