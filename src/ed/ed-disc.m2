-----------------------------------------------------------------------------
-- ED Discriminant Computation (Macaulay2)
--
-- Computes the defining polynomial of the ED Discriminant (evolute)
-- for a shallow neural network with quadratic activation.
-----------------------------------------------------------------------------

restart

-----------------------------------------------------------------------------
-- Function: EDdisc
--
-- Inputs:
--   W1, b1: Weights and biases for the first layer (Input -> Hidden)
--   W2, b2: Weights and biases for the second layer (Hidden -> Output)
--
-- Output:
--   An ideal in QQ[u1, u2] representing the data locus where the
--   critical points of the distance function coincide.
-----------------------------------------------------------------------------

EDdisc = (W1, W2, b1, b2) -> (
    R := QQ[x1, x2, lambda, u1, u2];

    -- 1. Define Decision Boundary F
    z := W1 * matrix{{x1}, {x2}} + b1;
    h := matrix {{z_(0,0)^2}, {z_(1,0)^2}};
    f := W2 * h + b2;
    F := f_(0,0) - f_(1,0);

    -- 2. ED System (Lagrange Multipliers)
    eqs := {F, x1 - u1 - lambda*diff(x1, F), x2 - u2 - lambda*diff(x2, F)};
    I := ideal eqs;

    -- 3. Compute Discriminant via Elimination
    -- Isolate derivatives w.r.t {x1, x2, lambda} (indices 0,1,2 in Ring)
    JacSub := (jacobian I)^{0,1,2};

    -- Add singularity condition (det = 0) and saturate out degenerate cases (rank < 2)
    Crit := I + ideal(det JacSub);
    Sat := saturate(Crit, minors(2, JacSub));

    D := eliminate(Sat, {x1, x2, lambda});

    return radical D;
)


------- EXAMPLES FROM THE PAPER -------

-- ED discriminant is a curve
W1 = matrix{{1,2},{3,1}};
W2 = matrix{{2,1},{1,2}};
b1 = matrix{{0},{1}};
b2 = matrix{{2},{1}};

EDdisc(W1, W2, b1, b2)


-- single point on the ED discriminant
W1 = matrix{{1,2},{3,1}};
W2 = matrix{{2,1},{1,2}};
b1 = matrix{{1},{2}};
b2 = matrix{{1},{1}};

EDdisc(W1, W2, b1, b2)
