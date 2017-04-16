function [ out ] = prep_rejectArtifactMAxMin( dat, varargin )

% dat = data;
opt = opt_cellToStruct(varargin{:});

% Need to check option default
% if isfield(opt.threshold)
%     warning('Please input the value of threshold');
% end

% if ~isfield(opt)
%     warning ('There is no threshold, Please input the value of threshold');
% end
%%
%Find the channel index
if ~isempty(opt.Ival)
    dat.x = dat.x(str2double(opt.Ival(1)):str2double(opt.Ival(2)),:,:);
end
    

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

for i =1 : size(selectData,3)
    valMaxTrial(:,i) = max(rejcrt(:,i));
end
if valMaxTrial <= str2double(opt.threshold)
    out.x = dat.x;
else
    for maxnum=1: size(selectData,3)

        if valMaxTrial(maxnum) > str2double(opt.threshold)
            inx = inx+1;
            rejectTrial(1,inx) = maxnum;
            rejectTrial(2,inx) = valMaxTrial(maxnum);
        end
    end
    ratio = round(size(rejArtifact,3)*(str2double(opt.ratio)/100));
    sortrejData = sort(rejectTrial(2,:));
    if length(rejectTrial)> ratio
        for num=1:size(sortrejData,2)
            A = find(sortrejData(num) == rejectTrial(2,:));
            if length(A) >= 2
                sortRejTrialInx(1,num) = rejectTrial(1,A(1));
                if sortRejTrialInx(num) == sortRejTrialInx(num-1)
                    sortRejTrialInx(1,num) = rejectTrial(1,A(2));
                end
            else
                sortRejTrialInx(1,num) = rejectTrial(1,A);
            end
        end
        sortRejTrial = [sortRejTrialInx;sortrejData];
        findRejTrial = sortRejTrial(:,(length(rejectTrial)-ratio)+1:size(rejectTrial,2));
        srtRejectTrial = sort(findRejTrial(1,:));
    else
        for num=1:size(sortrejData,2)
            sortRejTrialInx(1,num) = rejectTrial(1,find(sortrejData(num) == rejectTrial(2,:)));
        end
        sortRejTrial = [sortRejTrialInx;sortrejData];
        srtRejectTrial = sort(sortRejTrial(1,:));
    end
    
    TrialInx = [1:size(dat.x,2)];
    exctTrialIndex = setdiff(TrialInx,srtRejectTrial);
    for dataTrial = 1: size(exctTrialIndex,2)
        out.x(:,dataTrial,:)  = dat.x(:,exctTrialIndex(dataTrial),:);
    end
end




end


