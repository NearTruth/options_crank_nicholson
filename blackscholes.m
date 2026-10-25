% run_nonuniform_fun(0.3, 2, 0.2, 2, "put")  % this will produce oscillations
% run_nonuniform_fun(5, 2, 0.9, 2, "call");

% run_fun(0.1, 2, 0.15, 2, "put")
% run_fun(0.1, 2, 0.2, 2, "call")
% run_nonuniform_fun(0.1, 2, 0.2, 2, "put")
% disp(mapping(1, 1/2, 2, 8, 1))
run_nonuniform_fun(0.8, 100, 0.1, 0.25, "put")



function run_nonuniform_fun(sigma, strike, interest, T, option)
% (a_fun, b_fun, c_fun, fun)
    numgrids = 5;
    errg_center = zeros(1, numgrids);
    ax = 0;
    bx = 8 * strike;
    rannacher = 4;

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
        [u5, u1, u2, u3, u4, d, g] = nonuniform_ivp_fun(a_fun, b_fun, c_fun, f, ga, gb, gamma, T, numpoints - 1, numtime, gridx, rannacher);
        
        for k = 2:numpoints
            [call, put] = blsprice(gridx(k), strike, interest, T, sigma);
            if(option == "call")
                truef(k-1) = call;
            else
                truef(k-1) = put;
            end
        end
        
        draw_sol_graph(gridx, u1, u2, u3, u4, u5, truef, sprintf("centered, nx=%d, nt = %d, theta = 0.5", numpoints, numtime), strike_grid_num)
        % draw_greeks_graph(gridx, d, g, sprintf("delta, nx=%d, nt = %d, theta = %.1f", numpoints, numtime, theta), sprintf("gamma, nx=%d, nt = %d, theta = %.1f", numpoints, numtime, theta), strike_grid_num)
        errg_center = errorblackscholes(gridx(2:numpoints), strike_grid_num, u5, n, errg_center, T, strike, interest, sigma, option);

    end
    
    convg1 = getconvergence(nint, errg_center, numgrids);

    disp_uniform(nint, ntime, errg_center, convg1, numgrids)
end



function [u, u1, u2, u3, u4, d, g] = nonuniform_ivp_fun(p, q, r, f, ga, gb, gamma, T, n, nt, nonuniform_gridx, rannacher)  % theta = crank-nicholson with rannacher
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
    
    u_prev = zeros([n, 1]);
    u1 = zeros([n, 1]);
    s_prev = zeros([n, 1]);

    total_timesteps = rannacher + nt - (rannacher / 2);  % half timestep 
    % total_timesteps = nt;
    
    j = 1;
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


        [k1, k3] = make_boundary(p, q, r, nonuniform_gridx, n, curr_time);

        s = theta * ht * s_new + (1 - theta) * ht * s_prev;
        s_prev = s_new;
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

        s = s + additional;
        s1 = L * u_prev + s;
        u = K\s1;  % we want to solve this using PSOR
        

        % we get the greeks here
        % d first derivative
        % g second derivative

        % index = 1
        d(1) = 1;  % (u(2) - ga_new) / hx;
        g(1) = 1;  % (ga_new - 2 * u(1) + u(2)) / hx ^ 2;
        % index = n
        d(n) = 1;  % (gb_new - u(n-1)) / hx;
        g(n) = 1;  % (u(n-1) - 2 * u(n) + gb_new) / hx ^ 2;
        
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
        r_val(i, 1) = - r(nonuniform_gridx(i+1), curr_time);
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

    A = A2 + A1 + spdiags(r_val, 0, n, n);
end

function [l1, l3] = make_boundary(p, q, ~, nonuniform_gridx, n, curr_time)
    h_w = nonuniform_gridx(2) - nonuniform_gridx(1);
    h_e = nonuniform_gridx(3) - nonuniform_gridx(2);

    l1 = -p(nonuniform_gridx(2), curr_time) * 2 / (h_w * (h_e + h_w)) + q(nonuniform_gridx(2), curr_time) * h_e / (h_w * (h_e + h_w));
    

    h_w = nonuniform_gridx(n+1) - nonuniform_gridx(n);
    h_e = nonuniform_gridx(n+2) - nonuniform_gridx(n+1);

    l3 =  -p(nonuniform_gridx(n+1), curr_time) * 2 / (h_e * (h_e + h_w)) - q(nonuniform_gridx(n+1), curr_time) * h_w / (h_e * (h_e + h_w));
end


function [u_new, k] = PSOR(C, u, b, g, omega, epsilon)
    % we assume that C is a tridiagonal matrix
    k = 1;
    n = size(u);
    u_prev = u;
    u_new = zeros([n,1]);
    while(k < 10)
        % we treat the first and last iterations differently
        i = 1;
        u_new(i) = u_prev(i) + omega / C(i, i) * (b(i) - C(i, i+1) * u_prev(i+1) + C(i, i) * u_prev(i));
        u_new(i) = max([u_new(i), g(i)]);

        for i = 2:n-1
            u_new(i) = u_prev(i) + omega / C(i, i) * (b(i) - C(i, i-1) * u_new(i-1) - C(i, i+1) * u_prev(i+1) + C(i, i) * u_prev(i));
            u_new(i) = max([u_new(i), g(i)]);
        end
        i = n;
        u_new(i) = u_prev(i) + omega / C(i, i) * (b(i) - C(i, i-1) * u_new(i-1) + C(i, i) * u_prev(i));
        u_new(i) = max([u_new(i), g(i)]);
        
        if(stop(u_new, u_prev, epsilon))
            break;
        end

    end
end


function s = stop(u, u_prev, epsilon)
    epsilon = 1e-6;
    max_top = max(abs(u - u_prev));
    max_bottom = max([max(abs(u)), 1]);

    s = ((max_top / max_bottom) < epsilon);
end



function y = const0(~, ~)
    y = 0;
end

function y = const1(~, ~)
    y = 1;
end



function y = const(c, ~, ~)
    y = c;
end

function y = linx(c, x, ~)
    y = c * x;
end

function y = sq(c, x, ~)
    y = c^2 * x^2;
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





function [y1, y2, y3, yt] = fun6(x, t)
    y1 = exp(x * t);    y2 = t*exp(x * t); y3 = t*t*exp(x *t); yt = x* exp(x*t);
end


function errg = errorblackscholes(gridx, n, uvct, nn, errg, T, strike, interest, sigma, option)
    for i = 1:n-1
        px = gridx(i);
        [call, put] = blsprice(px, strike, interest, T, sigma);
        if(option == "call")
            truef = call;
        else
            truef = put;
        end

        errg(1, nn) = max(errg(1, nn), abs(truef-uvct(i)));
    end
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


function disp_uniform(nint, nts, errg, convg, ntimes)
    fprintf('   n   nt    errg  convg\n');
    nn = 1; fprintf('%4d %4d %9.4e   ----  \n', nint(nn), nts(nn), errg(:, nn));
    for nn = 2:ntimes
        fprintf('%4d %4d %9.4e %6.2f\n', nint(nn), nts(nn), errg(:, nn), convg(nn-1));
    end
end


function draw_sol_graph(fullgridx, full_u1, full_u2, full_u3, full_u4, full_u5, full_true, t, strike_grid_num)
    figure();
    nexttile
    gridx = fullgridx(1:strike_grid_num);
    u1 = full_u1(1:strike_grid_num);
    u2 = full_u2(1:strike_grid_num);
    u3 = full_u3(1:strike_grid_num);
    u4 = full_u4(1:strike_grid_num);
    u5 = full_u5(1:strike_grid_num);
    true = full_true(1:strike_grid_num);


    plot(gridx, u1, 'r-', gridx, u2, 'g-', gridx, u3, 'b-', gridx, u4, 'c-', gridx, u5, 'm-' , gridx, true, 'k-');
    title(t)
    hx = xlabel('x');
    hy = ylabel('sol');
    hl = legend("T = 0", "T = 0.25", "T = 0.5", "T = 0.75", "T = 1", "True");
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
