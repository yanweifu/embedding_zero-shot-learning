function [X] = normalize(X)

for i=1:size(X,1)
    X(i,:) = X(i,:)./norm(X(i,:));
end
