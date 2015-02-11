function matr = doMatrxZscore(input)
% first do transformation to normalize each row;
% then do transformation to normalize each column;

% first do row normalization; 
row = zscore(input');

% then do column normalization:

matr=zscore(row');