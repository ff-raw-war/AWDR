function [B, A, Z] = train(X, c, alpha, ep1, ep2, epsilon, maxIter)
    
    
    lossVec = [];
    iterVec = [];
    errVec = [];
    errVec2 = [];

    rho = 1.01; 
	% rho = 3;
    mu = 0.1; 
    maxMu = 10^5; 
    threshold = 2*10^-4; 
    [d, n] = size(X);
    dd = d;
    %{
    
    [anchor, ind2, score] = VDA(X', c);
   
    A = anchor'; 
    %｝
    
    
    scatter(X(:,1), X(:,2), 'b.') % Blue dots
    hold on
    scatter(anchor(:,1), anchor(:,2), 'r+') % Red plus signs
    %}
    B = rand(d,dd);
    Z = rand(c,n);
    W = ep2./(abs(Z)+ep1);
    A = rand(dd, c);
    J = rand(d,n); % Lagrange乘子
    es1 = ones(c, 1);
    es2 = ones(n, 1);

    if gpuDeviceCount >= 1
        X = gpuArray(X);
        B = gpuArray(B);
        Z = gpuArray(Z);
        W = gpuArray(W);
        J = gpuArray(J);
    end

    err = 10*threshold; 
    errVec = [errVec err];
    loss = getLoss(Z);
    lossVec = [lossVec ; loss]; 
    iter = 1;
    iterVec = [iterVec iter];
	notConverged = 1;
    % initialTime = toc
    while notConverged
        iter = iter + 1;
        loss0 = loss;

        % tic;
        B = getB(X, A, Z, J, mu);
        % timeP = toc
        
        W = ep2./(abs(Z)+ep1);
        Q = diag(0.5 ./ sqrt(sum(Z .* Z , 2) + eps));
        Z = getZ(X, B, A, W, Q, alpha, mu, Z);

        A = getA(X, B, Z, J, mu);
        % tic;

        % mu = ALF; 
        mu = min(rho*mu, maxMu); 
        
        
        err = norm(Z'*es1-es2, 'fro');
		% err = max(max(abs(P'*X-P'*X*C-E)));

		loss = getLoss(Z);
        errVec = [errVec err];
        lossVec = [lossVec loss];
        iterVec = [iterVec iter];
		
        
		if iter >= maxIter
			notConverged = 0;
		end
        
        
		if err<threshold
			notConverged = 0;
		end
        
		%{
        
        if ((loss0-loss)'*(loss0-loss) < epsilon) || (loss > loss0)
            notConverged = 0;
        end % 
        %}
        
        % afterTime = toc
    end % while iter
    
    %{
    
    %plot(hengzhou, zongzhou, 'd-');
    if iter < 10
        for i = 1:10-iter
            iterVec = [iterVec iter+i];
            lossVec = [lossVec min(lossVec)];
            errVec = [errVec err];
        end
    end
    plot(iterVec, lossVec, 'd-');
    title('AR');
    xlabel('Number of iterations');
    ylabel('Log of the objective value');
    %}

end
