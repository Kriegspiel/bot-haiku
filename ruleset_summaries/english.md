# English

Play ordinary chess on the true board with opponent pieces and moves hidden. Illegal moves are public referee announcements and must be retracted; the player keeps trying until a legal move stands. The prompt's recent scorecard turns carry public announcements and any private move details the server allows from your own perspective.

Legal captures announce that a capture occurred and give the capture square, but not the capturing man or captured man. Ordinary captures do not reveal pawn-vs-piece identity, so `pawns_captured` is absent. En passant is the exception: it is announced explicitly on the capturing pawn's destination square. Checks announce direction only. Promotion and castling are legal but not announced, and promoted piece type stays hidden.

`ask_any` asks whether any legal pawn capture exists. After a positive answer, the player must try one pawn capture. If that required pawn-capture try is illegal, the player is released and may finish with any legal move. Stalemate is an ordinary draw announcement, not RAND's stalemate-loss rule.
