%% SAIS FUNCTION
function [Pf_hat, T_final] = SAIS_Algorithm_new(Sfun, N, K, rho, dx, lambda, maxIter, seed)
% SAIS_ALGORITHM
% Core MATLAB implementation of the Subset Adaptive Importance Sampling (SAIS).
%
% Inputs:
%   Sfun     : limit state function handle, Sfun(x)
%   N        : number of proposal distributions
%   K        : samples per proposal
%   rho      : quantile parameter (e.g., 0.1)
%   dx       : dimensionality of the problem
%   lambda   : recycling forgetting factor in (0,1)
%   maxIter  : maximum number of SAIS iterations
%   seed     : (optional) random number generator seed
%
% Outputs:
%   Pf_hat   : final recycled estimate of the failure probability
%   T_final  : total number of iterations completed

    if nargin >= 8 && ~isempty(seed)
        rng(seed);
    end

    % ============================================================
    % INTERNAL INITIALIZATION 
    % ============================================================
    b_target = 0;
    t = 1;
    b_prev = inf;           
    I_hat = [];
    
    % 1. Initial means uniformly distributed in [-1, 1]^d_x (size: dx x N)
    mu = -1 + 2 * rand(dx, N);
    
    % 2. Initial covariances as isotropic identity matrices (size: dx x dx x N)
    Sigma = repmat(eye(dx), [1, 1, N]);

    while (b_prev >= b_target) && (t <= maxIter)
        % ============================================================
        % STEP 1 — Sampling and seed selection
        % ============================================================
        X = zeros(dx, N*K);
        proposal_id = zeros(1, N*K);
        
        col = 1;
        for n = 1:N
            % Force strict symmetry to avoid mvnrnd floating-point crashes
            Sig_n = (Sigma(:,:,n) + Sigma(:,:,n)') / 2;
            
            % Generate samples (Output is dx x K)
            Xn = mvnrnd(mu(:,n)', Sig_n, K)';   
            X(:, col:col+K-1) = Xn;
            proposal_id(col:col+K-1) = n;
            col = col + K;
        end
        
        NK = size(X, 2);
        
        % Evaluate performance function
        Svals = zeros(1, NK);
        for k = 1:NK
            Svals(k) = Sfun(X(:,k));
        end
        
        % Proposal-wise failure samples under previous subset F^(t-1) and elites
        A_all = [];
        for n = 1:N
            idx_n = find(proposal_id == n);
            idx_fail_n = idx_n(Svals(idx_n) <= b_prev);   % Samples inside F^(t-1)
            
            M_n = numel(idx_fail_n);
            A_n = floor(rho * M_n);
            
            if A_n >= 1
                % Sort ascending to grab the lowest S(x) values (closest to failure boundary)
                [~, ord] = sort(Svals(idx_fail_n), 'ascend');
                elite_idx = idx_fail_n(ord(1:A_n));
                A_all = [A_all, elite_idx];
            end
        end
        
        if isempty(A_all)
            warning('No elites found. Stopping algorithm early.');
            break;
        end

        % ============================================================
        % STEP 2 — Threshold adaptation
        % ============================================================
        S_elite = Svals(A_all);
        [S_sorted, ~] = sort(S_elite, 'descend'); 
        A_total = numel(S_sorted);
        
        idx_q = max(1, floor(rho * A_total));
        b = S_sorted(idx_q);                     % New threshold b^(t)
             
        % Current active subset F^(t)
        idx_active = find(Svals <= b);

        % ============================================================
        % STEP 3 — Proposal adaptation
        % ============================================================
        
        % Evaluate likelihoods of ALL samples under ALL N proposals
        qvals = zeros(N, NK);
        for n = 1:N
            Sig_n = (Sigma(:,:,n) + Sigma(:,:,n)') / 2;
            qvals(n,:) = mvnpdf(X', mu(:,n)', Sig_n)';
        end
        
        % DM weights: w_k = pi(x_k) / Psi(x_k), Psi = (1/N) sum_n q_n
        Psi = sum(qvals, 1) / N;
        pi_vals = mvnpdf(X', zeros(1,dx), eye(dx))';
        w = pi_vals ./ max(Psi, realmin);
        
        % Posterior probabilities delta_n(x_k) 
        % Reassignment is carried out efficiently on the active subset samples only
        delta_active = qvals(:, idx_active) ./ max(sum(qvals(:, idx_active), 1), realmin);
        [~, assign_active] = max(delta_active, [], 1);
        
        mu_old = mu;
        Sigma_old = Sigma;
        
        for n = 1:N
            idx_n = idx_active(assign_active == n);   % Reassigned samples for proposal n
            Kstar = numel(idx_n);
            
            % Safety check: Need at least dx + 1 samples to compute a valid covariance
            if Kstar < dx + 1
                continue; % Starved proposal: keep previous valid mean and covariance
            end
            
            Xn = X(:, idx_n);
            wn = w(idx_n);
            wn = wn / sum(wn);
            
            ESS = 1 / sum(wn.^2);
            NT = Kstar / 2;
            
            % Temper weights if ESS is too small
            if ESS < NT
                gamma_t = 1 / (1 + exp(-t));     % Sigmoid schedule from the paper
                wn = wn .^ gamma_t;
                wn = wn / sum(wn);
            end
            
            % Mean update (Eq. 13)
            mu_new = Xn * wn';
            
            % Covariance update (Eq. 14 or tempered equivalent)
            Xc = Xn - mu_new;
            Sigma_hat = Xc * diag(wn) * Xc';
            Sigma_hat = (Sigma_hat + Sigma_hat') / 2;
            
            % Ledoit-Wolf shrinkage coefficient beta^(t)
            beta_val = computeLW_centered(Xc, Sigma_hat);
            
            % Isotropic diagonal empirical covariance
            Sigma_tilde = (trace(Sigma_hat) / dx) * eye(dx);
            
            % eta^(t) = 0.1 * t^{-1}
            eta = 0.1 / t;
            
            % Eq. (17) Shrinkage Formula
            Sigma_new = (1 - beta_val) * Sigma_old(:,:,n) + beta_val * Sigma_hat + eta * Sigma_tilde;
            
            % Force strictly Symmetric Positive-Definite (SPD)
            Sigma_new = nearestSPD_with_floor(Sigma_new, 1e-8);
            
            mu(:,n) = mu_new;
            Sigma(:,:,n) = Sigma_new;
        end

        % ============================================================
        % STEP 4 — Failure estimation
        % ============================================================
        indicator = double(Svals <= 0);
        I_hat(t) = sum(w .* indicator) / NK;
        
        b_prev = b;
        t = t + 1;
    end

    % ============================================================
    % OUTPUT — Final Recycled Estimator
    % ============================================================
    T_final = numel(I_hat);
    if T_final == 0
        Pf_hat = NaN;
    else
        alpha_vec = lambda .^ (T_final - (1:T_final));
        alpha_norm = (1 - lambda) / (1 - lambda^T_final);
        Pf_hat = alpha_norm * sum(alpha_vec .* I_hat);
    end
end

% ========================================================================
% HELPER FUNCTION: Ledoit-Wolf coefficient using centered samples
% ========================================================================
function beta_val = computeLW_centered(Xc, S)
    % Xc : dx x K centered samples
    % S  : covariance estimate based on centered samples
    [dx, K] = size(Xc);
    num = 0;
    for k = 1:K
        diffk = Xc(:,k) * Xc(:,k)' - S;
        num = num + norm(diffk, 'fro')^2;
    end
    den = K^2 * (trace(S*S) - (trace(S)^2)/dx);
    if den <= 0 || ~isfinite(den)
        beta_val = 1;
    else
        beta_val = num / den;
        beta_val = min(max(beta_val, 0), 1);
    end
end

% ========================================================================
% HELPER FUNCTION: Force SPD with minimum eigenvalue floor
% ========================================================================
function A = nearestSPD_with_floor(A, epsFloor)
    A = (A + A') / 2;
    [V, D] = eig(A);
    d = diag(D);
    d(~isfinite(d)) = epsFloor;
    d = max(d, epsFloor);
    A = V * diag(d) * V';
    A = (A + A') / 2;
end