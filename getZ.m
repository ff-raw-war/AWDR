function Z = getZ(X, B, A, W, Q, alpha, mu, Z0)
    [c, n] = size(Z0);
    [d, n] = size(X);
    es1 = ones(c, 1);
    es2 = ones(n,1);
    % Z = pinv(2*eye(c)+mu1*A'*B'*B*A+mu2*es1*es1')*(A'*B'*J+mu1*A'*B'*X+mu2*es1*es2');
    Z = (alpha*W(1:c,1:c).*Q+2*A'*B'*B*A+mu*es1*es1')\(2*A'*B'*X+mu*es1*es2');
    Z(isnan(Z))=0;
end