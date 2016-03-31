function [ rejected_data ] = prep_rejectArtifactMAxMin( data, varargin )

dat = data;
opt = opt_cellToStruct(varargin{:});
% if isfield(opt.threshold)
%     warning('Please input the value of threshold');
% end

%Find the channel index
if ~isempty(opt.channel)
    if opt.channel{1} == ':'
        for chInx = 1: size(dat.chSet,2)
            selectChanInx(1,chInx) = chInx;
        end
    else
        for num=1: size(opt.channel,2)
            for chInx = 1: size(dat.chSet,2)
                if strcmp(opt.channel{num}, dat.chSet{chInx}) == 1
                    selectChanInx(1,num) = chInx;
                end
            end
        end     
    end
end

rejArtifact = permute(dat.x , [1,3,2]);
for chanSelect = 1 : size(selectChanInx,2)
    selectData(:,chanSelect,:) = rejArtifact(:,selectChanInx(chanSelect), :);
end
rejArtifactData = reshape(selectData, [size(selectData,1) , size(selectData,2)*size(selectData,3)]);

rejmax = max(rejArtifactData, [], 1);
rejmin = min(rejArtifactData, [], 1);
rejcrt = rejmax-rejmin;
rejcrt = reshape(rejcrt , [size(selectData,2) , size(selectData,3)]);
inx = 0;
for maxnum=1: size(selectData,3)
    valMaxTrial(1,maxnum) =max(rejcrt(:,maxnum)); 
    if valMaxTrial(maxnum) > str2double(opt.threshold)
        inx = inx+1;
        rejectTrial(1,inx) = maxnum;
        rejectTrial(2,inx) = valMaxTrial(maxnum);
    end
end
ratio = size(rejArtifact,3)*(str2double(opt.ratio)/100);
sortrejData = sort(rejectTrial(2,:));
if length(rejectTrial)> ratio
    for num=1:size(sortrejData,2)
        sortRejTrialInx(1,num) = rejectTrial(1,find(sortrejData(num) == rejectTrial(2,:)));
    end
    sortRejTrial = [sortRejTrialInx;sortrejData];
    findRejTrial = sortRejTrial(:,(length(rejectTrial)-ratio)+1:size(rejectTrial,2));
end
srtRejectTrial = sort(findRejTrial(1,:));
TrialInx = [1:size(dat.x,2)];
exctTrialIndex = setdiff(TrialInx,srtRejectTrial);
for dataTrial = 1: size(exctTrialIndex,2) 
    rejected_data(:,dataTrial,:)  = dat.x(:,exctTrialIndex(dataTrial),:);
end

% filteredData = dat.x(:,)

