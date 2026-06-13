%% ========================================================================
% iCE-IS GM CORE ALGORITHM (CLEAN & ANNOTATED)
% Based on: "Improved cross entropy-based importance sampling with a 
% flexible mixture model" (Papaioannou et al., 2019)
% ========================================================================
function [Pf_hat, T_final] = iCEIS_GM_core(Sfun, dim, k_clusters, ns, n_final, target_cv, max_iter)
    
    % --- INITIALIZATION ---
    % As per Section 3 (Improved cross entropy method), the algorithm is initialized.
    % In this specific implementation, instead of the vMFNM model, a Gaussian 
    % Mixture Model (GMM) is used as the parametric family.
    
    % Initialize smooth approximation parameter sigma_0 = infinity (Eq. 17 context)
    % Here initialized as a sufficiently large number (10.0) for standard normal space.
    sigma_prev = 10.0; 
    
    % Initialize GMM parameters (means, covariances, and weights)
    means = cell(k_clusters, 1);
    covs = cell(k_clusters, 1);
    weights = ones(k_clusters, 1) / k_clusters;
    
    % Start with a basic uniform prior (Step 1 of the iCE algorithm)
    for k = 1:k_clusters
        means{k} = normrnd(0, 0.2, [1, dim]);
        covs{k} = eye(dim);
    end
    
    % --- CROSS-ENTROPY ADAPTATION PHASE ---
    % This loop corresponds to Steps 2-5 of the iCE algorithm described in Section 3.
    for t = 1:max_iter
        
        % Safe normalize mixture weights
        weights = max(weights, 0);
        weights = weights / sum(weights);
        
        % Step 2: Generate samples {u_k} from the current IS density h(u, v_{t-1})
        u = zeros(ns, dim);
        idx_samples = randsample(k_clusters, ns, true, weights);
        for i = 1:ns
            u(i, :) = mvnrnd(means{idx_samples(i)}, covs{idx_samples(i)});
        end
        
        % Step 2 (cont.): Calculate the responses G(u_k)
        g_val = zeros(ns, 1);
        for i = 1:ns
            g_val(i) = Sfun(u(i, :)');
        end
        
        % ============================================================
        % CONVERGENCE CHECK (Step 3)
        % ============================================================
        % Evaluate the sample CV of the weights with respect to the optimal IS density.
        % Using the smooth approximation: h_t(u) proportional to \Phi(-G(u)/\sigma) (Eq. 15)
        
        % phi_prev corresponds to \Phi(-G(u_k)/\sigma_{t-1})
        phi_prev = normcdf(-g_val / sigma_prev);
        
        % w_stop evaluates the term: I(G(u_k) <= 0) / \Phi(-G(u_k)/\sigma_{t-1})
        w_stop = double(g_val <= 0) ./ max(phi_prev, 1e-15);
        
        % Calculate sample mean and standard deviation of these weights
        m_ws = mean(w_stop);
        if m_ws > 0
            cv_stop = std(w_stop) / m_ws;
        else
            cv_stop = inf;
        end
        
        % If the CV is smaller than \delta_{target} AND we have failure samples, stop.
        if cv_stop <= target_cv && any(g_val <= 0)
            T_final = t;
            break; % Go to Step 6
        end
        
        % ============================================================
        % SMOOTHING PARAMETER OPTIMIZATION (Step 4)
        % ============================================================
        % Solve Eq. (17) to determine \sigma_t such that the CV of the new weights 
        % equals the target CV (\delta_{target}).
        
        % Evaluate the nominal standard normal density \varphi_n(u) for all samples
        phi_n = mvnpdf(u, zeros(1, dim), eye(dim));
        
        % Evaluate the current sampling density h(u, \hat{v}_{t-1}) for all samples
        h_prev = zeros(ns, 1);
        for j = 1:k_clusters
            h_prev = h_prev + weights(j) * mvnpdf(u, means{j}, covs{j});
        end
        
        % Define the objective function for Eq. (17): 
        % (\delta_{W_t}(\sigma) - \delta_{target})^2
        % where W_t(u_i) = \eta_t(u_i) / h(u_i, \hat{v}_{t-1})
        % and \eta_t(u_i) = \Phi(-G(u_i)/\sigma) * \varphi_n(u_i)
        obj_sig = @(s) (std( (normcdf(-g_val / s) .* phi_n) ./ max(h_prev, 1e-18) ) / ...
                       (mean( (normcdf(-g_val / s) .* phi_n) ./ max(h_prev, 1e-18) ) + 1e-18) - target_cv)^2;
                   
        % Solve the 1D optimization problem for \sigma_t
        options = optimset('Display', 'off');
        sigma_t = fminbnd(obj_sig, 1e-5, sigma_prev, options);
        
        % ============================================================
        % PARAMETER UPDATE (Step 5)
        % ============================================================
        % Compute \hat{v}_t through solving the stochastic program of Eq. (16).
        % For a Gaussian Mixture, this corresponds to the weighted EM algorithm 
        % described in Section 5.
        
        % Calculate the intermediate weights W_t (Eq. 16)
        w_t = (normcdf(-g_val / sigma_t) .* phi_n) ./ max(h_prev, 1e-18);
        w_t = max(w_t, 0);
        w_t = w_t / sum(w_t); % Normalize weights for stability
        
        % Expectation-Maximization (EM) loop (Section 5)
        for em = 1:10
            % Expectation Step: Calculate \gamma_{i,j}^{(l)}
            gamma = zeros(ns, k_clusters);
            for j = 1:k_clusters
                gamma(:, j) = weights(j) * mvnpdf(u, means{j}, covs{j});
            end
            gamma = gamma ./ max(sum(gamma, 2), 1e-18);
            
            % Maximization Step: Update weights, means, and covariances
            sum_w_gamma = sum(w_t .* gamma, 1);
            for j = 1:k_clusters
                weights(j) = sum_w_gamma(j) / (sum(w_t) + 1e-18);
                if sum_w_gamma(j) > 1e-12
                    eff_weight = w_t .* gamma(:, j);
                    
                    % Update Mean
                    means{j} = sum(eff_weight .* u, 1) / sum_w_gamma(j);
                    
                    % Update Covariance
                    diff_u = u - means{j};
                    covs{j} = (diff_u' * (diff_u .* eff_weight)) / sum_w_gamma(j) + 1e-5 * eye(dim);
                end
            end
        end
        
        % Store \sigma_t for the next iteration
        sigma_prev = sigma_t;
        T_final = t; 
    end
    
    % ============================================================
    % FINAL ESTIMATION PHASE (Step 6)
    % ============================================================
    % Generate n_f samples from the final adapted IS density h(u, \hat{v}_T)
    % and estimate P_F.
    
    u_final = zeros(n_final, dim);
    idx_final = randsample(k_clusters, n_final, true, weights);
    for i = 1:n_final
        u_final(i, :) = mvnrnd(means{idx_final(i)}, covs{idx_final(i)});
    end
    
    % Calculate responses G(u_k) for the final samples
    g_final = zeros(n_final, 1);
    for i = 1:n_final
        g_final(i) = Sfun(u_final(i, :)');
    end
    
    % Evaluate nominal density \varphi_n(u)
    phi_n_f = mvnpdf(u_final, zeros(1, dim), eye(dim));
    
    % Evaluate final IS density h(u, \hat{v}_T)
    h_f = zeros(n_final, 1);
    for j = 1:k_clusters
        h_f = h_f + weights(j) * mvnpdf(u_final, means{j}, covs{j});
    end
    
    % Calculate the standard IS weights: W(u_k, \hat{v}_T) = \varphi_n(u_k) / h(u_k, \hat{v}_T)
    % and multiply by the indicator function I(G(u_k) <= 0) (Eq. 4)
    w_final_eval = (double(g_final <= 0) .* phi_n_f) ./ max(h_f, 1e-18);
    
    % The final estimate \hat{P}_F is the sample mean of these weights
    Pf_hat = mean(w_final_eval);
end