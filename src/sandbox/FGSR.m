%% FGSR.m
% Author:      Joshua Eckels
% Description: Class to provide helper functions for non-uniform structured
% grid
%              
% Calling:    FGSR.function_name(arg1,arg2)
% Date:       1/17/21

classdef FGSR
    methods(Static)
        function [sg, yg] = compute_grid(A,b,c,ymin,ymax)
            N = 100;
            param = [ymin,N,A,b,c,ymax];
            R = @(vguess)FGSR.residual(vguess,param);    % Residual
            dRdvg = @(vguess)FGSR.deriv(vguess,R);       % Residual derivative
            toggle = 0;
            maxIter = 1000;
            tol = 10^(-6);
            vguess0 = 0.01; % initial guess for shooting parameter dy/ds(s=0)
            
            % Solve for shooting paramter with Newton-Raphson
            [vfinal, n_iter] = FGSR.newton_raphson(R,dRdvg,vguess0,tol,maxIter,toggle);
            
            % Use rk4 to compute the y-direction mesh from s=0 to s=1
            v0 = [ymin;vfinal];                     % Initial conditions
            dvds = @(s,v_vec)FGSR.state_deriv(s,v_vec,param);
            ds = 1/(N-1);
            [sg, v_mat] = FGSR.rk4(dvds, v0, ds, N-1); % Time step state system
            yg = v_mat(:,1);
        end

        % Derivative function for state system
        % Parameters:
        %   s     - location value to evaluate dvds at
        %   v_vec - state vector to evaluate dvds at
        % Outputs:
        %   dvds  - the output vector of derivatives of state vector
        % Calling:
        %   @(s,v_vec)state_deriv(s, v_vec); params are passed in
        function dvds = state_deriv(s, v_vec,param)
            % parameters passed in
            Ny = param(2);  % Number of y points
            A = param(3);   % Tuning parameters
            b = param(4);   %
            c1 = param(5);  %
            
            f1 = (A/2)*(1-erf(b*(s-c1)));
        %     f2 = (A/2)*(1-erf(b*(s-c2)));
        %     fGSR = f1 + f2 - A;             % Grid-stretching function
             
            n = size(v_vec,1); % v_vec comes in as column state vector
            dvds = zeros(n,1); % dvds will leave as column state vector
            dvds(1) = v_vec(2);
            dvds(2) = (Ny-1)*f1*v_vec(2);
        end
        
        % Newton-Raphson
        % Parameters:
        %   R - residual function; R(x)
        %   dRdx - derivative of residual function; dRdx(x)
        %   xi - initial guess
        %   tol - tolerance
        %   maxIter - number of iterations maximum
        %   toggle - 0; no print info. 1; print info
        function [soln, n_iter] = newton_raphson(R, dRdx, xi, tol, maxIter, toggle)
            corr = abs(R(xi)/dRdx(xi));
            xi_new = xi;
            n = 0; % number of iterations
            if (toggle == 1)
                fprintf('%5s %7s %9s %9s %9s\n','Count','xi','R(xi)','dRdx(xi)','corr');
            end
            
            while (abs(corr) > tol)
                n = n + 1;
                Rxi = R(xi);
                dRdxi = dRdx(xi);
                corr = -Rxi/dRdxi;
                xi_new = xi + corr;
                if (toggle == 1)
                    fprintf('%5.0f %7.3f %9.3f %9.3f %9.3f\n', n, xi, Rxi, dRdxi, corr);
                end
                xi = xi_new;
                if (n >= maxIter)
                    break;
                end
            end
            % found solution (or exceeded maxIter)
            soln = xi_new;
            n_iter = n;
        end
        
        % Computes the residual of grid-stretching second-order DE for vguess
        % Parameters:
        %   vguess - independent variable for shooting parameter
        % Calling:
        %   residual expected to be called as R(vguess); params are passed in
        function res = residual(vguess,param)
            % parameters passed in
            ymin = param(1);        % initial condition [m]
            N = param(2);           % Number of points
            ymax = param(6);        % Boundary Condition for y at s=1
            
            % Run rk4 to time-step state vector to s=1 using vguess
            v0 = [ymin;vguess];    % Initial conditions
            dvds = @(s,v_vec)FGSR.state_deriv(s,v_vec,param);
            ds = 1/(N-1);
            [s, v_mat] = FGSR.rk4(dvds, v0, ds, N-1); % Time step state system
            
            % Pull out end results from rk4 (at edge of computational domain s=1)
            yend = v_mat(end,1);
            dydsend = v_mat(end,2);
           
            % Compute residual
            res = yend - ymax;
        end
        
        % computes an approximate derivative of a residual function
        % Parameters:
        %   R - function handle to approximate derivative of
        %   x - location to evaluate the derivative
        % Calling:
        % deriv expected to be called as dRdx(x); so R function handle must be
        % passed in
        function d = deriv(x,R)
            d = (R(1.01*x) - R(x))/(0.01*x);
        end
        
        % Runge-Kutta approximation with 4 slopes
        % Parameters:
        %   dv_fcn - function handle which evaluates derivative of state vector
        %   v0_fcn - vector with initial value of each state variable [nx1]
        %   dt     - time step 
        %   N      - number of time steps
        % Outputs:
        %   t_vec  - vector of time values [ (N+1) x 1 ]
        %   v_mat  - matrix with instantaneous value of each state variable 
        %            [ (N+1) x n ]
        % Calling:
        %   dv_fcn expected to be called as dv_fcn(t, v_vec), where v_vec is the
        %   current state vector
        function [t_vec, v_mat] = rk4(dv_fcn, v0_vec, dt, N)
            n = size(v0_vec,1);     % n = number of state variables
            t_vec = (0:dt:dt*N)';   % time column vector
            v_mat = zeros(N+1,n);   % columns for each state var; rows for time steps
            v_mat(1,:) = v0_vec';   % row=0; t=0; v = v0 (ICs)
            for i = 1:N
                k1 = dv_fcn(t_vec(i),v_mat(i,:)'); % d{v}/dt 1 (col vector)
                ystar1 = v_mat(i,:)' + (dt/2)*k1;
                
                k2 = dv_fcn(t_vec(i) + dt/2, ystar1); % d{v}/dt 2
                ystar2 = v_mat(i,:)' + (dt/2)*k2;
                
                k3 = dv_fcn(t_vec(i) + dt/2, ystar2); % d{v}/dt 3
                ystar3 = v_mat(i,:)' + dt*k3;
                
                k4 = dv_fcn(t_vec(i) + dt, ystar3); % d{v}/dt 4
                v_mat(i+1,:) = v_mat(i,:) + dt*(k1' + 2*k2' + 2*k3' + k4')/6;
            end
        end
    end
end