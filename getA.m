function A = getA(X, B, Z, J, mu1)
    % A = (mu1*B'*B)\(B'*J*Z'+mu1*B'*X*Z')/(Z*Z');
    % A = pinv(mu1*B'*B)*(B'*J*Z'+mu1*B'*X*Z')*pinv(Z*Z');
    BXZ=B'*X*Z'; % d'×c   PXZ' = d'd dn nc
    BXZ(isnan(BXZ))=0;
    BXZ(isinf(BXZ))=max(BXZ(isfinite(BXZ)));
    BXZ(BXZ==-inf)=min(BXZ(isfinite(BXZ)));
    [U,~,V]=svd(BXZ,'econ');
    A = U*V';
end