run_nonuniform_PSOR(0.2, 2, 0.01, 0.5, "put")  % this will produce oscillations
% run_nonuniform_fun(0.3, 2, 0.9, 2, "put");

% run_fun(0.1, 2, 0.15, 2, "put")
% run_fun(0.1, 2, 0.2, 2, "call")
% run_nonuniform_fun(0.1, 2, 0.2, 2, "put")
% disp(mapping(1, 1/2, 2, 8, 1))




function run_nonuniform_PSOR(sigma, strike, interest, T, option)
% (a_fun, b_fun, c_fun, fun)
    numgrids = 6;
    errg_center = zeros(1, numgrids);
    ax = 0;
    bx = 8 * strike;
    rannacher = 0;

    f = @const0;

    a_fun = @(x, t) 0.5 * sigma * sigma * x * x;
    b_fun = @(x, t) interest * x;
    c_fun = @(x, t) -interest;

    if(option == "call")
        ga = @(t) 0;  % call option
        gb = @(t) call_boundary(strike, T, interest, bx, t);
        gamma = @(x) call_fun(strike, x, 0);
    else
        ga = @(t) put_boundary(strike, T, interest, 0, t);  % call option
        gb = @(t) 0;
        gamma = @(x) put_fun(strike, x, 0);

    end

    % mapping function
    % determine the coefficients
    
    alpha = 0.375;
    beta = get_beta(alpha, bx, strike);
    map_f = @(x) mapping(x, alpha, beta, bx, strike);

    for j = 1:numgrids
        numpoints = 2^(j+2);
        nint(j) = numpoints;
        numtime = 2^(j+2);
        ntime(j) = numtime;
    end
    
    u_val = zeros([numgrids, 1]);
    total_iter = zeros([numgrids, 1]);
    u_prev = zeros([4, 1]);
    for n = 1:numgrids
        
        numpoints = nint(n);
        numtime = ntime(n);
        % numpoints = 10;
        hx = (bx-ax)/(numpoints);
        gridx = ax + hx * [0:numpoints];
        strike_grid_num = numpoints-1;  % ceil((1.5 * strike) / hx); 
        truef = zeros(numpoints-1);
        gridx = map_f(gridx);
        
        % for i = 1:numpoints-1
        %     truefx(i) = gfunc(func, ax + i * hx, T);
        %     fx(i) = f(ax + i * hx, 0);
        % end
        

        % add rannacher smoothing
        [u5, u1, u2, u3, u4, d, g, PSOR_epsilon, total_iter(n)] = nonuniform_ivp_fun(a_fun, b_fun, c_fun, f, ga, gb, gamma, T, numpoints - 1, numtime, gridx, rannacher);
        
        
        draw_sol_graph(gridx, u1, u2, u3, u4, u5, sprintf("centered, nx=%d, nt = %d, theta = 0.5", numpoints, numtime), strike_grid_num);
        draw_greeks_graph(gridx, d, g, sprintf("delta, nx=%d, nt = %d", numpoints, numtime), sprintf("gamma, nx=%d, nt = %d", numpoints, numtime), strike_grid_num);
        [errg_center, u_val(n)] = errorblackscholes(numpoints, u5, u_prev, n, errg_center);
        
        disp(PSOR_epsilon);
        u_prev = u5;
    end
    
    convg1 = getconvergence(nint, errg_center, numgrids);

    disp_nonuniform(nint, ntime, u_val, errg_center, convg1, numgrids, total_iter)
end



function [u, u1, u2, u3, u4, d, g, PSOR_epsilon, total_iterations] = nonuniform_ivp_fun(p, q, r, f, ga, gb, gamma, T, n, nt, nonuniform_gridx, rannacher)  % theta = crank-nicholson with rannacher
% p, q, r coeff functions depends on (x, t)
% f = ut - (pu_xx + qu_x + ru) is given 
% g boundary condition function
% [a, b] interval of definition
% T time interval 
% nt number of time points to consider
% n number of all points, total points = n+1
% nonuniform_gridx the nonuniform gridpoints to consider
% rannacher the number of rannacher timesteps to run
    
    ht_ref = T / nt;
    
    d = zeros([n, 1]);
    g = zeros([n, 1]);

    payoff = zeros([n, 1]);
    
    u_prev = zeros([n, 1]);
    u1 = zeros([n, 1]);
    s_prev = zeros([n, 1]);

    k_arr = zeros([nt, 1]);
    PSOR_epsilon = 1e-4;

    total_timesteps = rannacher + nt - (rannacher / 2);  % half timestep 
    % total_timesteps = nt;
    
    j = 1;
    total_iterations = 1;
    while (j <= total_timesteps)  % 4 rannacher steps and n-2 normal steps
        additional = zeros([n, 1]);
        s_new = zeros([n, 1]);
        

        % we check that the current j is below 4
        if(j <= rannacher)
            % we must change theta, ht
            theta = 1;
            ht = ht_ref /2;
            curr_time = j * ht;
        else
            theta = 1/2;
            ht = ht_ref;
            curr_time = (j- rannacher/2 ) * ht;  % half timestep
            % curr_time = j * ht;
        end

        if(j == 1)
            for i = 1:n
                u_prev(i, 1) = gamma(nonuniform_gridx(i+1));
                payoff(i, 1) = u_prev(i, 1);

                u1(i, 1) = u_prev(i, 1);
            end
            
            for i = 1:n
                s_prev(i, 1) = f(nonuniform_gridx(i+1), 0);
            end

            ga_prev = ga(0);
            gb_prev = gb(0);

            
            A = make_matrix(p, q, r, nonuniform_gridx, n, 0);
            
            [l1, l3] = make_boundary(p, q, r, nonuniform_gridx, n, 0);
            l1 = -l1;
            l3 = -l3;
        end

        for i = 1:n
            s_new(i, 1) = f(nonuniform_gridx(i+1), curr_time);
        end
        
        % A_new = make_matrix(p, q, r, nonuniform_gridx, n, curr_time);


        [k1, k3] = make_boundary(p, q, r, nonuniform_gridx, n, 0);

        s = theta * ht * s_new + (1 - theta) * ht * s_prev;
        s_prev = s_new;  % this is just 0
        ga_new = ga(curr_time);
        gb_new = gb(curr_time);


        additional(1, 1) =  ht * (- theta * k1 * ga_new + (1 - theta) * l1 * ga_prev);  % fix the timestepping of the boundaries
        additional(n, 1) =  ht * (- theta * k3 * gb_new + (1 - theta) * l3 * gb_prev);

        ga_prev = ga_new;
        gb_prev = gb_new;
        
        l1 = -k1;
        l3 = -k3;

        K = speye(n, n) + theta * ht * A; 
        L = speye(n, n) - (1- theta) * ht * A;
    


        % disp(full(L))
        % disp(full(L_alt))

        s = s + additional;
        s1 = L * u_prev + s;
        % u_alt = K\s1;  % we want to solve this using PSOR

        [u, k] = PSOR(K, u_prev, s1, payoff, 1.35, PSOR_epsilon);
        k_arr(j) = k;
        total_iterations = total_iterations + k;
        % disp(u - u_alt);
        % disp(k);

        
        for i = 2:n-1
            d(i) = 1;  % (u(i+1) - u(i-1)) / hx;
            g(i) = 1;  % (u(i-1) - 2 * u(i) + u(i+1)) / hx ^ 2;
        end

        % u = [g(a); u; g(b)];  % append the end points
        

        if ((j-2)/nt == 1/4)
            for i = 1:n
                u2(i, 1) = u(i, 1);
            end
        end
        if ((j-2)/nt == 1/2)
            for i = 1:n
                u3(i, 1) = u(i, 1);
            end
        end
        if ((j-2)/nt == 3/4)
            for i = 1:n
                u4(i, 1) = u(i, 1);
            end
        end
        u_prev = u;

        j = j+1;
    end
    [d, g] = get_delta_gamma(ga_new, gb_new, u, nonuniform_gridx);
end


function A = make_matrix(p, q, r, nonuniform_gridx, n, curr_time)
    p_val = zeros([n,1]);
    q_val = zeros([n,1]);
    r_val = zeros([n,1]);
    
    delta1 = zeros([n,1]);  % if we store these we can reuse them
    delta2 = zeros([n,1]);
    delta3 = zeros([n,1]);

    gamma1 = zeros([n,1]);
    gamma2 = zeros([n,1]);
    gamma3 = zeros([n,1]);
    % temp2(i, 1) = (2 * p(nonuniform_gridx(i+1), curr_time) * ht / (h_w * h_e) - q(nonuniform_gridx(i+1), curr_time) * (h_e^2 - h_w^2) * ht / denom - r(nonuniform_gridx(i+1), curr_time) * ht);
    for i = 1:n
        h_w = nonuniform_gridx(i+1) - nonuniform_gridx(i);
        h_e = nonuniform_gridx(i+2) - nonuniform_gridx(i+1);
        denom = h_e * h_w * (h_e + h_w);
        delta1(i, 1) =  q(nonuniform_gridx(i+1), curr_time) * (h_e * h_e) / denom;
        delta2(i, 1) = - q(nonuniform_gridx(i+1), curr_time) * (h_e^2 - h_w^2) / denom;
        delta3(i, 1) = - q(nonuniform_gridx(i+1), curr_time) * (h_w * h_w) / denom;
    
        gamma1(i, 1) = - p(nonuniform_gridx(i+1), curr_time) * (2 * h_e) / denom;
        gamma2(i, 1) = 2 * p(nonuniform_gridx(i+1), curr_time) / (h_w * h_e);
        gamma3(i, 1) = - p(nonuniform_gridx(i+1), curr_time) * (2 * h_w) / denom;
        
        % p_val(i, 1) = p(nonuniform_gridx(i+1), curr_time);
        % q_val(i, 1) = q(nonuniform_gridx(i+1), curr_time);
        r_val(i, 1) = r(nonuniform_gridx(i+1), curr_time);
    end 

    A2 = spdiags(gamma2, 0, n, n);
    A2 = spdiags(gamma1(2:n), -1, A2);
    diag = [0; gamma3(1:n-1)];
    A2 = spdiags(diag, 1, A2);

    A1 = spdiags(delta2, 0, n, n);
    A1 = spdiags(delta1(2:n), -1, A1);
    diag = [0; delta3(1:n-1)];
    A1 = spdiags(diag, 1, A1);

    A0 = speye(n, n);

    A = A2 + A1 - spdiags(r_val, 0, n, n);
end

function [l1, l3] = make_boundary(p, q, ~, nonuniform_gridx, n, curr_time)
    h_w = nonuniform_gridx(2) - nonuniform_gridx(1);
    h_e = nonuniform_gridx(3) - nonuniform_gridx(2);

    l1 = -p(nonuniform_gridx(2), curr_time) * 2 / (h_w * (h_e + h_w)) + q(nonuniform_gridx(2), curr_time) * h_e / (h_w * (h_e + h_w));
    

    h_w = nonuniform_gridx(n+1) - nonuniform_gridx(n);
    h_e = nonuniform_gridx(n+2) - nonuniform_gridx(n+1);

    l3 =  -p(nonuniform_gridx(n+1), curr_time) * 2 / (h_e * (h_e + h_w)) - q(nonuniform_gridx(n+1), curr_time) * h_w / (h_e * (h_e + h_w));
end

function [d, g] = get_delta_gamma(ga, gb, u, nonuniform_gridx)
    n = size(u, 1);
    d = zeros([n, 1]);
    g = zeros([n, 1]);
    h_w = nonuniform_gridx(2) - nonuniform_gridx(1);
    h_e = nonuniform_gridx(3) - nonuniform_gridx(2);
    denom = h_e * h_w * (h_e + h_w);
    % index = 1
    d(1) = (-ga * (h_e * h_e) + u(1) * (h_e^2 - h_w^2) + u(2) * (h_w * h_w) )/ denom;
    g(1) = (ga * (2 * h_e) - 2 * u(1) * (h_e + h_w) + u(2) * (2 * h_w)) / denom;
    
    % index = n
    
    h_w = nonuniform_gridx(n+1) - nonuniform_gridx(n);
    h_e = nonuniform_gridx(n+2) - nonuniform_gridx(n+1);
    denom = h_e * h_w * (h_e + h_w);
    % index = 1
    d(n) = (-u(n-1) * (h_e * h_e) + u(n) * (h_e^2 - h_w^2) + gb * (h_w * h_w) )/ denom;
    g(n) = (u(n-1) * (2 * h_e) - 2 *u(n)* (h_e + h_w) + gb * (2 * h_w)) / denom;
    
    for i = 2:n-1
        h_w = nonuniform_gridx(i+1) - nonuniform_gridx(i);
        h_e = nonuniform_gridx(i+2) - nonuniform_gridx(i+1);
        denom = h_e * h_w * (h_e + h_w);
        % index = 1
        d(i) = (-u(i-1) * (h_e * h_e) + u(i) * (h_e^2 - h_w^2) + u(i+1) * (h_w * h_w) )/ denom;
        g(i) = (u(i-1) * (2 * h_e) - 2 *u(i)* (h_e + h_w) + u(i+1) * (2 * h_w)) / denom;
    end
end


function [u_new, k] = PSOR(C, u, b, g, omega, epsilon)
    % we assume that C is a tridiagonal matrix
    k = 1;
    n = size(u, 1);

    u_prev = u;
    u_new = zeros([n,1]);
    while(k < 100)
        % we treat the first and last iterations differently
        i = 1;
        u_new(i) = u_prev(i) + (omega / C(i, i)) * (b(i) - C(i, i+1) * u_prev(i+1) - C(i, i) * u_prev(i));
        u_new(i) = max([u_new(i), g(i)]);

        for i = 2:n-1
            u_new(i) = u_prev(i) + (omega / C(i, i)) * (b(i) - C(i, i-1) * u_new(i-1) - C(i, i+1) * u_prev(i+1) - C(i, i) * u_prev(i));
            u_new(i) = max([u_new(i), g(i)]);
        end
        i = n;
        u_new(i) = u_prev(i) + (omega / C(i, i)) * (b(i) - C(i, i-1) * u_new(i-1) - C(i, i) * u_prev(i));
        u_new(i) = max([u_new(i), g(i)]);
        
        if(stop(u_new, u_prev, epsilon))
            break;
        end

        % disp("new")
        % disp(u_new)
        % disp("prev")
        % disp(u_prev)
        % % disp(full(C))
        % disp(b)

        u_prev = u_new;
        k = k + 1;
    end
end


function s = stop1(u, u_prev, epsilon)
    s = (norm(u - u_prev) <= epsilon);
end

function s = stop(u, u_prev, epsilon)
    max_one = @(x) max([x, 1]);
    n = size(u, 1);
    max_arr = zeros([n, 1]);

    max_bottom = arrayfun(max_one, u);
    max_top = max(abs(u - u_prev) ./ max_bottom);

    % for i = 1:n
    %     max_arr(i) = abs(u(i) - u_prev(i)) / max_one(u(i));
    % end
    % 
    % max_top = max(max_arr);

    % disp("top")
    % disp(abs(u - u_prev))
    % disp("bottom")
    % disp(max_bottom)
    % disp(max_top);
    

    s = (max_top <= epsilon);
end



function y = const0(~, ~)
    y = 0;
end


function y = mapping(x, alpha, beta, bx, E)
    y = (1 + sinh(beta * (x / bx - alpha))/sinh(alpha * beta)) * E;
end

function y = get_beta(alpha, bx, E)
    f = @(beta) mapping(bx, alpha, beta, bx, E) - bx;
    y = fzero(f, 1);
end



function y = call_fun(strike, x, ~)
    y = max(x - strike, 0);
end


function y = put_fun(strike, x, ~)
    y = max(strike - x, 0);
end

function y = call_boundary(strike, T, r, x, t)
    y = x - strike * exp(-r * t);
end

function y = put_boundary(strike, T, r, ~, t)
    y = strike * exp(-r * t);
end



function [errg, u_strike] = errorblackscholes(n, uvct, u_prev, nn, errg)
    % we assume the relation of n_new = 2 * n_prev
    % we get the error at the strike
    n_prev = n / 2;
    alpha = 3 / 8;
    if(n_prev < 8)
        strike_prev = 0;
    else
        strike_prev = u_prev(alpha * n_prev);
    end
    
    strike_curr = uvct(alpha * n);

    u_strike = strike_curr; 
    % from prev point index to new point index is (2 * i)
    errg(1, nn) = max(errg(1, nn), abs(strike_prev-strike_curr));

end


function convg = getconvergence(nint, errg, ntimes)
    errg(:, :) = max(errg(:, :),  0.222044604925e-15);
    LogNintRatio = log(nint(1, 2:ntimes)./nint(1, 1:ntimes-1));
    LogNintRatioMat = repmat(LogNintRatio, size(errg, 1), 1);
    convg = 0;
    if ntimes > 1
        convg = log(errg(:, 1:ntimes-1)./errg(:, 2:ntimes))./LogNintRatioMat;
    end

end


function disp_nonuniform(nint, nts, u_val, errg, convg, ntimes, total_iterations)
    fprintf('   n   nt   u strike      errg    convg  iters   ratio\n');
    nn = 1; fprintf('%4d %4d %10.6f %9.4e   ----  %6.2f  ----\n', nint(nn), nts(nn), u_val(nn), errg(:, nn), total_iterations(nn));
    for nn = 2:ntimes
        fprintf('%4d %4d %10.6f %9.4e %6.2f  %6.2f  %6.2f\n', nint(nn), nts(nn), u_val(nn), errg(:, nn), convg(nn-1), total_iterations(nn), errg(nn-1)/errg(nn));
    end
end

function draw_sol_graph(fullgridx, full_u1, full_u2, full_u3, full_u4, full_u5, t, strike_grid_num)
    figure();
    nexttile
    gridx = fullgridx(1:strike_grid_num);
    u1 = full_u1(1:strike_grid_num);
    u2 = full_u2(1:strike_grid_num);
    u3 = full_u3(1:strike_grid_num);
    u4 = full_u4(1:strike_grid_num);
    u5 = full_u5(1:strike_grid_num);


    plot(gridx, u1, 'r-', gridx, u2, 'g-', gridx, u3, 'b-', gridx, u4, 'c-', gridx, u5, 'm-');
    title(t)
    hx = xlabel('x');
    hy = ylabel('sol');
    hl = legend("T = 0", "T = 0.25", "T = 0.5", "T = 0.75", "T = 1");
end


function draw_greeks_graph(fullgridx, full_d, full_g, t1, t2, strike_grid_num)
    nexttile
    gridx = fullgridx(1:strike_grid_num);
    d = full_d(1:strike_grid_num);
    g = full_g(1:strike_grid_num);


    plot(gridx, d, 'r-');
    title(t1)
    hx1 = xlabel('x');
    hy1 = ylabel('delta');
    hl = legend("delta");
    nexttile
    plot(gridx, g, 'g-');
    title(t2)
    hx2 = xlabel('x');
    hy2 = ylabel('gamma');
    hl = legend("gamma");
end
