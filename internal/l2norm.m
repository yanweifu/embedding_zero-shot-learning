function res = l2norm(X, coef)
% do l2-normalization for each row of the matrix X.

if nargin<2
    coef = 1;
end

sqsum= X.*X;
res = X./repmat(sqrt(sum(sqsum,2)./coef),1,size(X,2));