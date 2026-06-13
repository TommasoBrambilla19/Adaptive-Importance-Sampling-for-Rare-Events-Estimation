%% ========================================================================
% SUBSET SIMULATION ALGORITHM (SEQUENTIAL COMPONENT-WISE MMA)
% Strictly matches the provided pseudo-algorithm and text logic step-by-step
% ========================================================================
function [Pf_hat, total_levels, total_evals] = Sequential_MMA_SuS_core(Sfun, dim, N, p0, sigma, max_levels)
    
    total_evals = 0;
    P_intermediate = zeros(max_levels, 1);
    
    % -------------------------------------------------------------------------
    % LEVEL 0: Initial Random Sampling (Standard Monte Carlo Phase)
    % -------------------------------------------------------------------------
    fprintf('  ---> [SuS Level 0] Drawing %d initial random samples...\n', N);
    U_curr = randn(N, dim); % Draw random samples from the standard normal target distribution
    g_curr = zeros(N, 1);   % This vector will hold the QoI value for each sample
    
    for i = 1:N
        g_curr(i) = Sfun(U_curr(i, :)'); % Compute the QoI value for each random sample
        total_evals = total_evals + 1;
    end
    
    Nc = floor(N * p0); % This is N_M.ch (the number of elite seeds we pick per level)
    Ns = ceil(N / Nc);  % This is 1/p (the number of samples we must generate from each seed)
    total_levels = 0;
    
    % -------------------------------------------------------------------------
    % LEVEL L > 0: Population Evolution Phase
    % -------------------------------------------------------------------------
    for lvl = 1:max_levels
        total_levels = lvl;
        
        % Sort all the current samples based on their QoI values
        [g_sort, sort_idx] = sort(g_curr, 'ascend');
        
        % Text Logic: Set intermediate boundary b(t) as the average between 
        % the last bad sample Q_(N_M.ch) and the first good sample Q_(N_M.ch + 1)
        b_target = 0.5 * (g_sort(Nc) + g_sort(Nc+1));
        
        % Line 7 text check: Check if any random samples already fall into the ultimate failure region (g <= 0)
        if b_target <= 0
            b_target = 0;
            num_fail = sum(g_curr <= 0); % Count how many fall into the ultimate failure region
            P_intermediate(lvl) = num_fail / N;
            break;
        end
        
        P_intermediate(lvl) = p0;
        
        % Extract the N_M.ch seeds found in this level to start the chains
        elite_idx = sort_idx(1:Nc);
        U_chain_curr = U_curr(elite_idx, :); % These are our starting seeds eta
        g_chain_curr = g_curr(elite_idx);    % These are their corresponding QoI values
        
        U_next = zeros(N, dim);
        g_next = zeros(N, 1);
        
        % The first step of each chain is the seed sample itself
        U_next(1:Nc, :) = U_chain_curr;
        g_next(1:Nc)    = g_chain_curr;
        
        % Loop to produce 1/p samples for each of the N_M.ch seeds
        for step = 2:Ns
            for c = 1:Nc
                % Identify the current sample state eta^(n) and its QoI value Q^(n)
                eta_curr = U_chain_curr(c, :);
                g_curr_state = g_chain_curr(c);
                
                % Prepare a temporary candidate placeholder eta_tilde
                eta_cand = eta_curr; 
                
                % =============================================================
                % PSEUDO-ALGORITHM STEP 1: Generate candidate sample component by component
                % =============================================================
                % Pseudo-code Line 2: FOR j = 1 TO M (Loop over each dimension)
                for j = 1:dim
                    
                    % Pseudo-code Line 3: Draw candidate component xi_j from proposal PDF 
                    % centered around the current component coordinate
                    xi_j = eta_curr(j) + randn * sigma;
                    
                    % Pseudo-code Line 4: Compute the ratio r_j = pi_j(xi_j) / pi_j(eta_j)
                    % Using the mathematical log-trick explained above for safety
                    log_rj = -0.5 * (xi_j^2 - eta_curr(j)^2);
                    
                    % Pseudo-code Line 5 & 6: Draw u from Uniform[0,1] and check if u <= min(r_j, 1)
                    if log(rand) <= log_rj
                        % Pseudo-code Line 7: Candidate component is accepted -> eta_tilde_j = xi_j
                        eta_cand(j) = xi_j; 
                    else
                        % Pseudo-code Line 9: Candidate component is rejected -> eta_tilde_j = eta_j
                        eta_cand(j) = eta_curr(j);
                    end
                % Pseudo-code Line 11: ENDFOR
                end 
                
                % =============================================================
                % PSEUDO-ALGORITHM STEP 2: Accept or reject the fully assembled sample
                % =============================================================
                % Compute the QoI value Q(eta_tilde) for the fully compiled candidate vector
                g_cand = Sfun(eta_cand');
                total_evals = total_evals + 1;
                
                % Pseudo-code Line 14: IF Q(eta_tilde) falls into the current failure region F_i
                if g_cand <= b_target
                    
                    % Pseudo-code Line 16: Accept the new sample -> eta^(n+1) = eta_tilde
                    U_chain_curr(c, :) = eta_cand;
                    
                    % Pseudo-code Line 17: Update current QoI state -> Q^(n+1) = Q(eta_tilde)
                    g_chain_curr(c) = g_cand;
                    
                else % Pseudo-code Line 18: ELSE
                    
                    % Pseudo-code Line 20: Reject new sample and retain old state -> eta^(n+1) = eta^(n)
                    U_chain_curr(c, :) = eta_curr;
                    
                    % Pseudo-code Line 21: Retain old QoI value -> Q^(n+1) = Q^(n)
                    
                % Pseudo-code Line 22: ENDIF
                end
                
                % Save this final resulting state into the main population matrix
                idx = (step - 1) * Nc + c;
                if idx <= N
                    U_next(idx, :) = U_chain_curr(c, :);
                    g_next(idx)    = g_chain_curr(c);
                end
            end
        end
        
        % Update the population pool variables to pass into the next level loop
        U_curr = U_next;
        g_curr = g_next;
    end
    
    % Compute the estimated rare-event probability: p_F = p^L * (N_fail / N_L)
    Pf_hat = prod(P_intermediate(1:total_levels));
end