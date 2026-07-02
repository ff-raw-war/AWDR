function [B, A, Z] = train(X, c, alpha, ep1, ep2, epsilon, maxIter)
    % tic
    
    lossVec = [];
    iterVec = [];
    errVec = [];
    errVec2 = [];

    rho = 1.01; % 原ADMM
	% rho = 3; % L0LRSSC方法
    mu = 0.1; % 原ADMM和L0LRSSC方法一样，都是0.1
    maxMu = 10^5; % L0LRSSC方法是10^6， SSC是10^5
    threshold = 2*10^-4; 
    [d, n] = size(X);
    dd = d; % 降维之后的维度
    %{
    % 用VDA来初始化A
    [anchor, ind2, score] = VDA(X', c);
    % c×d, 选取下标, n×c
    A = anchor'; % A是d×c
    %｝
    
    % 对锚点画图
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
    lossVec = [lossVec ; loss]; % 记录每一次迭代的loss的值，从第0次开始
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

        % mu = ALF; % SSC的mu的算法，是固定值
        mu = min(rho*mu, maxMu); % GMC-LRSSC方法
        
        % 终止迭代条件为约束项
        err = norm(Z'*es1-es2, 'fro');
		% err = max(max(abs(P'*X-P'*X*C-E)));

		loss = getLoss(Z);
        errVec = [errVec err];
        lossVec = [lossVec loss];
        iterVec = [iterVec iter]; % 记录迭代次数，从第0此开始
		% 检查是否收敛
        
		if iter >= maxIter
			notConverged = 0;
		end
        
        % 约束项收敛
		if err<threshold
			notConverged = 0;
		end
        
		%{
        % 终止迭代条件为loss值差
        if ((loss0-loss)'*(loss0-loss) < epsilon) || (loss > loss0)
            notConverged = 0;
        end % 
        %}
        
        % afterTime = toc
    end % while iter
    
    %{
    % 画图用
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