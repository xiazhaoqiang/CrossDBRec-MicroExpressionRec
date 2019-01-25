% Compare different models
clear all;

srcNet = load(fullfile('model','imagenet-resnet-50-dag.mat'));
tarNet = stresnetInit([220 200]);

Nlayers = numel(srcNet.layers);
fprintf('Checking...\n');
for i = 1:Nlayers
    % name
    status = strcmp(srcNet.layers(i).name,tarNet.layers(i).name);
    if status ~= 1
        fprintf('The name of %d-th layer is different.\n',i);
    end
    % input
    Nin_src = numel(srcNet.layers(i).inputs);
    Nin_tar = numel(tarNet.layers(i).inputs);
    if Nin_src == Nin_tar
        for j = 1:Nin_src
            status = strcmp(srcNet.layers(i).inputs{j},tarNet.layers(i).inputs{j});
            if status ~= 1
                fprintf('The inputs of %d-th layer are different.\n',i);
                break;
            end
        end
    else
        fprintf('The inputs number of %d-th layer are different.\n',i);
    end
    % output
    Nout_src = numel(srcNet.layers(i).outputs);
    Nout_tar = numel(tarNet.layers(i).outputs);
    if Nout_src == Nout_tar
        for j = 1:Nout_src
            status = strcmp(srcNet.layers(i).outputs{j},tarNet.layers(i).outputs{j});
            if status ~= 1
                fprintf('The outputs of %d-th layer are different.\n',i);
                break;
            end
        end
    else
        fprintf('The outputs number of %d-th layer are different.\n',i);
    end
end