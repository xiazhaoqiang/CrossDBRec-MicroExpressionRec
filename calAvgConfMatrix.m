function confMatrixM = calAvgConfMatrix(confMatrixT)
% CALAVGCONFMATRIX calculates the average confusion matrix

% Copyright (C) 2018 Zhaoqiang Xia.
% All rights reserved.

[rows,cols,deps] = size(confMatrixT);
confMatrixM = zeros(rows,cols);
for i = 1:rows
    for j = 1:cols
        confMatrixM(i,j) = mean(confMatrixT(i,j,confMatrixT(i,j,:)~= 0));
    end
end

end