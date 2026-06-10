# CrazyKrieg

CrazyKrieg combines hidden-board Kriegspiel with Crazyhouse reserves. Both players' reserves are public, and `allowed_moves` is authoritative for normal moves and reserve drops. Illegal normal moves and illegal drops are public rejections; the mover keeps trying until a legal move or legal drop stands. The attempted square is known only from your own scorecard perspective if the prompt includes it.

Legal captures announce the square and exact reserve identity of the captured unit: pawn, knight, bishop, rook, or queen. A promoted pawn enters reserve as a pawn and is announced as a pawn if captured. Legal moves or drops that give check announce direction only; the checking piece square or drop square is not announced.

`ask_any` checks visible pawns for legal captures. After a positive answer, one pawn-capture try is required; if that required try is illegal, the player may finish with any legal move. Pawn drops are separate from pawn captures and are legal only when present in `allowed_moves`. Promotion is legal but not announced when it happens, and promoted piece type stays hidden. Stalemate is an ordinary draw announcement, not RAND's stalemate-loss rule.
