function B = getB(X, A, Z, J, mu1)
    % B = (J*Z'*A'+mu1*X*Z'*A')*pinv(mu1*A*Z*Z'*A');
    XZA = X * Z' * A'; %d×d'  A*Z*X'=d'×d
    XZA(isnan(XZA))=0;
    [U, ~, V] = svd(XZA, 'econ'); % 用TIP2023的办法来求B
    B = U * V';
end