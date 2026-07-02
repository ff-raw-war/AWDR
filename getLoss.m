function loss = getLoss(Z)
    % 目标函数是：

    % 在计算loss的时候，不用管s.t.约束项
    


    loss = norm(Z,"fro")^2;
    
end