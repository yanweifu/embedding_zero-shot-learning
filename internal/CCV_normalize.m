function matrix =CCV_normalize(matrix,flag)
% input:
%       matrix: feature vectors for CCV
%       flag = 1,normalize each row
%       flag = 2,normalize each column,
%       flag = 3,normalize the matrix by the maximum and minimum
%       flag = 4,normalize the matrix by floating maximum (99.9%)
% output:
%       X: normalized feature vectors
%
%
% Note that: We must consider some video missing all
% the features.(e.g. some video does not have MFCC features)

% eps: smoothing factor.
eps=1e-30; 

if flag ==1
    % normalize each row:
    row_sum=sum(matrix,2);
    row_div_sum =repmat(row_sum,1,size(matrix,2));
    matrix = matrix./(row_div_sum+eps);    
end

if flag ==2
    % normalize each column:
    col_sum=sum(matrix,1);
    col_div_sum = repmat(col_sum,size(matrix,1),1);
    matrix = matrix./(col_div_sum+eps);
end

if flag ==3
    % normalize the whole matrix by the minimum and maximum number.
    Max_num = max(matrix(:));
    Min_num = min(matrix(:));
    if Max_num~=0;
        matrix = (matrix-Min_num)./(Max_num-Min_num);
    end
end

if flag ==4
    % float upbound(maximum value) of feature_matrix
    max_perc=0.999;
    bin_No = hist(matrix(:),[min(matrix(:)):max(matrix(:))]);
    threshold = sum(bin_No)*max_perc;
    sum_cnt = 0;
    for i=1:length(bin_No)
        if sum_cnt>=threshold
            Max_num_bin = i;
            fprintf('Maximum bin is %d\n', i);
            break;
        else
            sum_cnt = sum_cnt + bin_No(i);
        end
    end
    
    matrix = (matrix-min(matrix(:)))./(Max_num_bin - min(matrix(:)));
    
    bigger = matrix>=1;
    matrix = matrix -matrix.*bigger+bigger;
    
end

    


