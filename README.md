# European and American options pricing solver
European and American options pricing solver using Crank-Nicholson method. 
All methods add Rannacher smoothing and a nonuniform grid to improve errors around the discontinuity.

# Files
blackscholes.m implements a pricing solver for European options.
blackscholes_penalty.m implements a pricing solver for American options using the penalty method.
blackscholes_american_sor.m implements a pricing solver for American options using an SOR method.

# Running
All files can be run directly. 
