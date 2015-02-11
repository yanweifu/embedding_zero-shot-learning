function  [dbgK] = filterKnn(K0, opts)
%
% parameters need to set in opts:
% opts.k
% opts.prototypeTime
% no = opts.no;
%
no = opts.no;
k = opts.k;

dN = size(K0,1);
    Kn = zeros(dN, dN);
    for j = 1:no
        % collect the k-nearest neighbors
        [~, indx] = sort(K0(j,:), 'descend');
        ind = indx(2:k+1);
        % only store the k-nearest neighbors in the similarity matrix
        Kn(j, ind) = K0(j, ind);
    end;
    
    
    for j = 1+no:dN
        % collect the k-nearest neighbors
        [~, indx] = sort(K0(j,:), 'descend');
       ind = indx(2:opts.prototypeTime*k+1);
    %    ind = indx(2:end);
        % only store the 3*k-nearest neighbors in the similarity matrix
        Kn(j, ind) = K0(j, ind);
    end;
    % compute the final symmetric similarity matrix
    dbgK = (Kn+Kn')/2; 
