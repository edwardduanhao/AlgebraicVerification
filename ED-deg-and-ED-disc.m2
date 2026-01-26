-----------------------------------------------------------------------------
-- ED Degree and Discriminant Computation
--
-- Purpose:
--   Computes the algebraic complexity of a shallow neural network with 
--   quadratic activation.
--
-- Steps:
--   1. Calculates the Euclidean Distance (ED) degree (the number of critical 
--      points for generic data).
--   2. Computes the ED discriminant (the locus of data points where 
--      critical points coincide).
-----------------------------------------------------------------------------


restart 

-----------------------------------------------------------------------------
-- Function: EDdegree
-- 
-- Purpose: 
--   Computes the Euclidean Distance (ED) degree of the decision boundary 
--   for a shallow neural network with quadratic activation.
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






-----------------------------------------------------------------------------
-- Function: EDdisc
-- Purpose: Computes the defining polynomial of the ED Discriminant (evolute)
--          for a shallow quadratic network.
-- Output:  An ideal in QQ[u1, u2] representing the data locus where the
--          critical points of the distance function coincide.
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


EDdegree(W1, W2, b1, b2) -- 4
EDdisc(W1, W2, b1, b2)





-- single point on the ED discriminant
W1 = matrix{{1,2},{3,1}};
W2 = matrix{{2,1},{1,2}};
b1 = matrix{{1},{2}};
b2 = matrix{{1},{1}};



EDdegree(W1, W2, b1, b2) -- 2
EDdisc(W1, W2, b1, b2)





