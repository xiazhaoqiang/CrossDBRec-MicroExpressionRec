% Test the deep model of RCNN by subjects
clear all;

% Settings
sizeType = 'size220';
imageSize = [220 200];
alpha_te = 8;
metaPath = fullfile('data','Annotation4crossdb.mat');
load(metaPath); % load 'crossdb'

% Load the testing data
imdbPath = fullfile('data',sizeType,['crossdb_mat_' num2str(alpha_te) '.mat']);
load(imdbPath); % load 'imdb'
% imdb.data = normalizeSubject(imdb.data,imdb.id); % normalize data
imdb.data = normalizeCrossDB(imdb.data); % normalize data
imdb.labels = imdb.labels + 1;
% 
subIdx = unique(crossdb.id);

preLabelsT = [];
gtLabelsT = [];
% each subject
for i = 1:numel(subIdx)
    load(fullfile('model',['strcn_obj_' num2str(i) '.mat']),'netStruct');
    net = dagnn.DagNN.loadobj(netStruct);
    fprintf('Testing...\n');
    index_te = find(imdb.id == subIdx(i));
    [perf,Labels] = rcnnTest(net,imdb.data(:,:,:,index_te),imdb.labels(index_te));
    confMatrix = calConfusionMatrix(double(Labels),imdb.labels(index_te), unique(imdb.labels));
    fprintf('Accuracy is %.3f.\n',perf);
    
    perfT(i) = perf;
    ConfMatrixT(:,:,i) = confMatrix;
    preLabelsT = cat(1,preLabelsT,Labels);
    gtLabelsT = cat(1,gtLabelsT,imdb.labels(index_te));
end
% Evaluate the performance
[avgAcc,Acc] = calBalancedAccuracy(preLabelsT,gtLabelsT);
[avgFscore,Fscores] = calF1Score(preLabelsT,gtLabelsT);
% % Save results
% currenttime = datevec(now);
% timestamp = [ num2str(currenttime(2)) '_' num2str(currenttime(3)) '_' num2str(currenttime(4)) '_' num2str(currenttime(5))];
% fprintf('The model has been saved.\n');
% save(fullfile('data',['results_' timestamp '.mat']),'avgAcc','Acc','avgFscore','Fscores','preLabelsT','gtLabelsT');