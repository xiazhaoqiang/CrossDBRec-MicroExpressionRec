function [perf,predLabels] = rcnnTest( net,testData,testLabel)
%RCNNTRAIN train RCNN dag models
%   net - trained net
%   testData - testing data
%   testLabel - testing labels

% Copyright (C) 2018 Zhaoqiang Xia.
% All rights reserved.

% swich to the gpu
if(gpuDeviceCount()>0)
    net.move('gpu') ;
else
    printf('No gpu!\n'); return;
end

N = size(testData,4);
batchSize = 20.0;
numBatches = ceil(N/batchSize);
predLabels = zeros(N,1);

for j = 0:numBatches-1
    % random select a minibatch
    ixTe = (1+j*batchSize):min((j+1)*batchSize,N);
    
    % load and preprocess images in a batch
    im = testData(:,:,:,ixTe);
    im_ = single(im); % note: 0-255 range
%     im_ = im_ - single(repmat(net.meta.normalization.averageImage,1,1,1,size(im_,4)));
%     averageImage = mean(im_,2);
%     im_ = im_ - single(repmat(averageImage,1,size(im_,2),1,1));
    im_ = gpuArray(im_);
    
    % validation
    net.mode = 'test' ;
    net.eval({'input',im_}) ;
    predLabels(ixTe,1) = predictLabels(net);
end
perf = numel(find(predLabels == testLabel))/numel(testLabel);

end

% -------------------------------------------------------------------------
function predLabels = predictLabels(net)

probs = squeeze(gather(net.vars(net.getVarIndex('prediction')).value)) ;
[p,predLabels] = max(probs,[],1);
predLabels = predLabels';

end