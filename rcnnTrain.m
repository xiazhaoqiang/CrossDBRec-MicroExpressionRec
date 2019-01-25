function net = rcnnTrain( net,trainData,trainLabel, varargin )
%RCNNTRAIN train RCNN dag models
%   net - initialized net
%   trainData - training data
%   trainLabel - training labels
%   varargin:
%   iters - iteration times
%   lr - initial learning rate

% Copyright (C) 2018 Zhaoqiang Xia.
% All rights reserved.

% swich to the gpu
if(gpuDeviceCount()>0)
    net.move('gpu') ;
    % initialize the momentum
    state.momentum = num2cell(zeros(1, numel(net.params))) ;
    state.momentum = cellfun(@gpuArray, state.momentum, 'uniformoutput', false) ;
else
    printf('No gpu!\n'); return;
end

if nargin < 4
    params.iterNum = 1;
    params.learningRate = 1e-2;
    batchSize = 10.0;
elseif nargin == 4
    params.iterNum = varargin{1};
    params.learningRate = 1e-2;
    batchSize = 10.0;
elseif nargin == 5
    params.iterNum = varargin{1};
    params.learningRate = varargin{2};
    batchSize = 10.0;
else
    params.iterNum = varargin{1};
    params.learningRate = varargin{2};
    batchSize = varargin{3};
end

parserv = [] ;
params.weightDecay = 0.0005;
params.momentum = 0.9;
params.lrDecay = 0.8;

N = size(trainData,4);

for i = 1:params.iterNum
    index = randperm(N) ;
    numBatches = ceil(N/batchSize);
    % split the val set
    ixVal = index((1+(numBatches-1)*batchSize):min(numBatches*batchSize,N));
    imVal = single(trainData(:,:,:,ixVal));
%     imVal = imVal - single(repmat(net.meta.normalization.averageImage,1,1,1,size(imVal,4)));
%     averageImage = mean(imVal,2);
%     imVal = imVal - single(repmat(averageImage,1,size(imVal,2),1,1));
    imVal = gpuArray(imVal);
    labelVal = gpuArray(trainLabel(ixVal));
    for j = 0:numBatches-1
        % random select a minibatch
        batch_time = tic;
        ixTr = index((1+j*batchSize):min((j+1)*batchSize,N));
        
        % load and preprocess images in a batch
        im = trainData(:,:,:,ixTr);
        im_ = single(im); % note: 0-255 range
%         % minus the global average
%         im_ = im_ - single(repmat(net.meta.normalization.averageImage,1,1,1,size(im_,4)));
        % minus the local average
%         averageImage = mean(im_,2);
%         im_ = im_ - single(repmat(averageImage,1,size(im_,2),1,1));
        % move to GPU
        im_ = gpuArray(im_);
        label_ = gpuArray(trainLabel(ixTr));
        
        % train the RCNN model
        net.mode = 'normal' ;
        net.eval({'input',im_,'label',label_}, {'objective',1}) ;
        % update
        state = update(net, state, params, numel(ixTr), parserv) ; % Note: numel(ixTr) = real batchsize
        batch_time = toc(batch_time) ;
        fprintf(' Iter %d  batch %d/%d (%.2f images/s) ,lr is %.1e\n', i, j+1,numBatches, numel(ixTr)/ batch_time,params.learningRate);
        
        % validation
        net.mode = 'test' ;
        net.eval({'input',im_,'label',label_}) ;
        fprintf('\t Train objective is %5f;',net.getVar('objective').value);
        net.mode = 'test' ;
        net.eval({'input',imVal,'label',labelVal}) ;
        fprintf('\t Val objective is %5f.\n',net.getVar('objective').value);
    end
    params.learningRate = params.learningRate*params.lrDecay;
end
net.reset() ;
net.move('cpu') ;
end

% -------------------------------------------------------------------------
% Accumulate gradient
% -------------------------------------------------------------------------
function state = update(net, state, params, batchSize, parserv)

for p = 1:numel(net.params)
    
    parDer = net.params(p).der ;

    switch net.params(p).trainMethod
        
        case 'average' % mainly for batch normalization
            thisLR = net.params(p).learningRate ;
            net.params(p).value = vl_taccum(...
                1 - thisLR, net.params(p).value, ...
                (thisLR/batchSize/net.params(p).fanout),  parDer) ;
            
        case 'gradient'
            thisDecay = params.weightDecay * net.params(p).weightDecay ;
            thisLR = params.learningRate * net.params(p).learningRate ;
            
            if thisLR>0 || thisDecay>0
                % Normalize gradient and incorporate weight decay.
                parDer = vl_taccum(1/batchSize, parDer, ...
                    thisDecay, net.params(p).value) ;
                
                % Update momentum.
                state.momentum{p} = vl_taccum(...
                    params.momentum, state.momentum{p}, ...
                    -1, parDer) ;
                
%                 delta = state.momentum{p} ;
                delta = vl_taccum(...
                    params.momentum, state.momentum{p}, ...
                    -1, parDer) ;
                % Update parameters.
                net.params(p).value = vl_taccum(1, net.params(p).value, thisLR, delta) ;
            end
        otherwise
            error('Unknown training method ''%s'' for parameter ''%s''.', ...
                net.params(p).trainMethod, ...
                net.params(p).name) ;
    end
end
end