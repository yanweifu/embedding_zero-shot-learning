function M = confusion_matrix(N,Ltrue,Lest)
% function M = confusion_matrix(N,Ltrue,Lest)
% N: Number of labels.
% Ltrue: true labels.
% Lest : estimated labels.
% M: row is the estimated labels; column is the true labels;
%

%Ensure vectors.
Ltrue = Ltrue(:); 
Lest  = Lest(:);

M = zeros(N,N);
if(min(Ltrue)==0), 
    Ltrue = Ltrue+1;
    Lest  = Lest+1;
end
for i = 1 : N
    for j = 1 : N
        M(i,j) = sum((Ltrue==i)&(Lest==j))/sum(Ltrue==i);
    end
end