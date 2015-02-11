function D=dist_mat(P1, P2)
%
% Euclidian distances between vectors
% P1 = double(P1);
% P2 = double(P2);
% 
% X1=repmat(sum(P1.^2,2),[1 size(P2,1)]);
% X2=repmat(sum(P2.^2,2),[1 size(P1,1)]);
% R=P1*P2';
% D=sqrt(X1+X2'-2*R);


D = -2*P1*P2';

n1 = zeros(1,size(P1,1));
n2 = zeros(1,size(P2,1));
for i=1:size(P1,1)
    n1(i) = norm(P1(i,:));
end
for i=1:size(P2,1)
    n2(i) = norm(P2(i,:));
end

for i=1:size(P1,1)
    for j=1:size(P2,1)
        D(i,j) = D(i,j) + n1(i)^2 + n2(j)^2;
    end
end

D = sqrt(D);







