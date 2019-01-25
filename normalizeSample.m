function dataNorm = normalizeSample(data)
%NORMALIZESAMPLE normalize the max and min values of individual samples
%   data: H*W*C*N, e.g., C = 2, N = 441
%   id:   N*1

% Copyright (C) 2018 Zhaoqiang Xia.
% All rights reserved.

% mag = sqrt(data(:,:,1,:).^2+data(:,:,2,:).^2);
% dataNorm = bsxfun(@rdivide, data, max(mag,1e-8));

[rows,cols,Nc,N] = size(data);

minValue = min(data,[],[1 2]);
minValue = repmat(minValue,[rows cols 1 1]);
% minValue = min(data,[],[1 2 3]);
% minValue = repmat(minValue,[rows cols Nc 1]);
maxValue = max(data,[],[1 2]);
maxValue = repmat(maxValue,[rows cols 1 1]);
% maxValue = max(data,[],[1 2 3]);
% maxValue = repmat(maxValue,[rows cols Nc 1]);
dataNorm = (data - minValue)./max(maxValue-minValue,1e-8);

end

