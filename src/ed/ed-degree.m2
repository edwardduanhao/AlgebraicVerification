-----------------------------------------------------------------------------
-- ED Degree Computation (Macaulay2)
--
-- Computes the Euclidean Distance (ED) degree of the decision boundary
-- for a shallow neural network with quadratic activation.
-----------------------------------------------------------------------------

restart

-----------------------------------------------------------------------------
-- Function: EDdegree
--
-- Inputs:
--   W1, b1: Weights and biases for the first layer (Input -> Hidden)
--   W2, b2: Weights and biases for the second layer (Hidden -> Output)
--
-- Output:
--   Integer representing the number of critical points of the distance
--   function from a generic data point to the decision boundary.
-----------------------------------------------------------------------------

EDdegree = (W1, W2, b1, b2) -> (
    -- 1. Setup Ring
    R := QQ[x1, x2];
    x := matrix{{x1}, {x2}};

    -- 2. Define Network & Decision Boundary F
    -- Calculate z = W1*x + b1
    z := W1 * x + b1;

    -- Square activation
    z1 := z_(0,0);
    z2 := z_(1,0);

    -- Output layer: f = W2 * z^2 + b2
    -- We manually construct the squared vector
    h := matrix{{z1^2}, {z2^2}};
    f := W2 * h + b2;

    -- Decision Boundary
    F := f_(0,0) - f_(1,0);

    -- 3. ED Degree Setup (Polar Variety)
    -- Random generic point u
    u := apply(2, i -> random(-100, 100));

    -- 4. Critical Point Matrix
    -- Column 1: Displacement (x_i - u_i)
    -- Column 2: Gradient (diff(x_i, F))
    M := matrix {
        { x1 - u#0,  diff(x1, F) },
        { x2 - u#1,  diff(x2, F) }
    };

    -- 5. Compute Ideal
    -- Condition: det(M) = 0 (vectors are parallel) AND point is on boundary (F)
    crit := ideal(det M) + ideal F;

    -- 6. Saturate
    -- Remove trivial solutions where the gradient vanishes
    grads := ideal(diff(x1, F), diff(x2, F));
    Sat := saturate(crit, grads);

    return degree Sat;
)


------- EXAMPLES FROM THE PAPER -------

-- ED discriminant is a curve
W1 = matrix{{1,2},{3,1}};
W2 = matrix{{2,1},{1,2}};
b1 = matrix{{0},{1}};
b2 = matrix{{2},{1}};

EDdegree(W1, W2, b1, b2) -- 4


-- single point on the ED discriminant
W1 = matrix{{1,2},{3,1}};
W2 = matrix{{2,1},{1,2}};
b1 = matrix{{1},{2}};
b2 = matrix{{1},{1}};

EDdegree(W1, W2, b1, b2) -- 2
