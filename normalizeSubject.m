function dataNorm = normalizeSubject(data,id)
%NORMALIZESUBJECT normalize the max and min values of individual subjects
%   data: H*W*C*N, here C = 2, N = 441
%   id:   N*1;

% Copyright (C) 2018 Zhaoqiang Xia.
% All rights reserved.

dataNorm = data;
idIdx = unique(id);
Nd  = numel(idIdx);
Nc = size(data,3);

% deal with each subject: 68 subjects
for s = 1:Nd
    range = find(id == idIdx(s));
    
%     minValue = min(data(:,:,:,range),[],'all');
%     maxValue = max(data(:,:,:,range),[],'all');
%     dataNorm(:,:,:,range) = (data(:,:,:,range) - minValue)/max(maxValue-minValue,1e-8);
    for i = 1:Nc
        minValue = min(data(:,:,i,range),[],'all');
        maxValue = max(data(:,:,i,range),[],'all');
        dataNorm(:,:,i,range) = (data(:,:,i,range) - minValue)/max(maxValue-minValue,1e-8);
    end

end

end

