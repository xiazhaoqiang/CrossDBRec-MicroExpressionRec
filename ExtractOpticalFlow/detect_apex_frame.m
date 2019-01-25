function [apexFrameIdx,onsetFrameIdx] = detect_apex_frame(fileList)
%DETECT_APEX_FRAME detect the apex frame by evaluating temporal changes
%   fileList - file list by 'dir' function

K = numel(fileList);
imgTensor = [];
shiftTensor = [];
for k = 1:K
    imgPath = fullfile(fileList(k).folder,fileList(k).name);
    I = double(rgb2gray(imread(imgPath)));
    imgTensor(:,:,k) = I;
end
shiftTensor = repmat(imgTensor(:,:,1),[1 1 K]);
sumT = std(imgTensor - shiftTensor,0,[1 2]);
sumF = squeeze(sumT);
[mValue,mIdx] = max(sumF);

if isempty(mIdx)
    mIdx = round(K/2);
end

onsetFrameIdx = 1;
apexFrameIdx = mIdx;
end

