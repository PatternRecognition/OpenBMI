function mrk= mrkutil_assignGivenLabelsToOnlineData(mrko, targetList, varargin)
% extraction of Target and Non-target stimuli in an online oddball task,
% specifying the targets 
% INPUT: 
% mrko          marker structure (e.g. mrk_orig )
% targetList    list specifying the target stimulus for each trial.
%               length(targetList) has to be equal to the number of trials 
%               (i.e. number of 'beginOfTrial-Markers')!
% varargin    
%     beginOfTrial    marker of a  befin of trial (default 50). Might also
%                     be a list of numbers in case there were inconsistencies
%     targetShift     shift of a non-target to target. E.e if a stimulus is 
%                     marked with '4' as a non-target and marked as 24 as 
%                     target, targetShift must be specified as 20!
%                     (default 10)
%     classDef        target and Nontarget specification
%                     default: {[11:19], [1:9]; 'Target', 'Non-target'},...
%     miscDef         further marked parameters
%                     default: {'S100', 'S251', 'S254'; 'cue off', 'start', 'end'},...
%                         
% OUTPUT
% mrk         updated marker structure with specified targets
% EXAMPLE
% JohannesHoehne 01/2011
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'targetShift', 10, ...
                       'classDef', {[11:19], [1:9]; 'Target', 'Non-target'},...
                       'miscDef', {'S100', 'S251', 'S254'; 'cue off', 'start', 'end'},...
                       'beginOfTrial', [50]);
                   
n = length(mrko.desc);
trialLength = zeros(1,length(targetList));
mrk_cp = mrko;
%modify the mark_cp...
iBlock=0;
currTarget = nan;
for ii=1:n
    currMrk = sscanf(char(mrk_cp.desc(ii)), '%*c%f%*c%f%*c%f',[1 Inf]);
    if sum(currMrk == unique(opt.beginOfTrial)) % is currMrk a 'beginOfTrial'
        iBlock=iBlock + 1; %next trial --> new target
        if iBlock > length(targetList)
            error('number of Trials in targetList doest match with the number of trials in the markers!!')
            return;
        end
        currTarget = targetList(iBlock);
    end
    if isnumeric(currMrk)
        if iBlock>0
            trialLength(iBlock) = trialLength(iBlock)+1 ;
        end
        if currMrk == currTarget
            nSpacesToAdd = length(mrko.desc{ii}) - length(['S' num2str(currMrk + opt.targetShift)]);
            switch nSpacesToAdd
                case 0
                    buf = '';
                case 1
                    buf = ' ';
                case 2
                    buf = '  ';
                case 3 %should't be happening ... !!
                    buf = '   ';
            end
            mrk_cp.desc{ii} = ['S' buf num2str(currMrk + opt.targetShift)];
        end
    end
end

if iBlock < length(targetList)
    error('number of Trials in targetList doest match with the number of trials in the markers!! \n length(targetList) = %i, number of trials = %i', length(targetList), iBlock)
    return;
end
%%check for errors
if (length(unique(trialLength)) == 1) && (trialLength(1) > 0)
    sprintf('the marker-conversion found %i markers with each trial having %i subtrials', n, trialLength(1))
elseif length(unique(trialLength)) > 1
    warning('the marker-conversion found %i markers, but the trials have a differing number of subtrials. This might be also due to unrelevant markers! \n number of subtrials per trial: \t %s \n occurances: \t \t \t %s', n, num2str(unique(trialLength)), num2str(histc(trialLength, unique(trialLength))))
elseif trialLength(1) == 0
    warning('The marker-conversion couldnt find any targets!!!')
end
mrk= mrk_defineClasses(mrk_cp, opt.classDef);




