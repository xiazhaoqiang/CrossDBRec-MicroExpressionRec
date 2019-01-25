function dataNorm = normalizeDataset(data)
%NORMALIZEDATASET normalize the max and min values of individual datasets
%   data: H*W*C*N, here C = 2, N = 441

% Copyright (C) 2018 Zhaoqiang Xia.
% All rights reserved.

dataNorm = data;
Nc  = size(data,3);

% % deal with CASME II
% range = 1:145;
% minValue = min(data(:,:,:,range),[],'all');
% maxValue = max(data(:,:,:,range),[],'all');
% dataNorm(:,:,:,range) = (data(:,:,:,range) - minValue)/max(maxValue-minValue,1e-8);
% 
% % deal with SMIC
% range = 146:309;
% minValue = min(data(:,:,:,range),[],'all');
% maxValue = max(data(:,:,:,range),[],'all');
% dataNorm(:,:,:,range) = (data(:,:,:,range) - minValue)/max(maxValue-minValue,1e-8);
% % deal with SAMM
% range = 310:441;
% minValue = min(data(:,:,:,range),[],'all');
% maxValue = max(data(:,:,:,range),[],'all');
% dataNorm(:,:,:,range) = (data(:,:,:,range) - minValue)/max(maxValue-minValue,1e-8);

% deal with CASME II
range = 1:145;
for i = 1:Nc
    minValue = min(data(:,:,i,range),[],'all');
    maxValue = max(data(:,:,i,range),[],'all');
    dataNorm(:,:,i,range) = (data(:,:,i,range) - minValue)/(maxValue-minValue);
end
% deal with SMIC
range = 146:309;
for i = 1:Nc
    minValue = min(data(:,:,i,range),[],'all');
    maxValue = max(data(:,:,i,range),[],'all');
    dataNorm(:,:,i,range) = (data(:,:,i,range) - minValue)/(maxValue-minValue);
end
% deal with SAMM
range = 310:441;
for i = 1:Nc
    minValue = min(data(:,:,i,range),[],'all');
    maxValue = max(data(:,:,i,range),[],'all');
    dataNorm(:,:,i,range) = (data(:,:,i,range) - minValue)/(maxValue-minValue);
end

end

