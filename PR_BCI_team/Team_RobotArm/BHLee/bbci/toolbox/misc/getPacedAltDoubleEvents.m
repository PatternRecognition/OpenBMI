function mrk_paced = getPacedAltDoubleEvents(mrk, mrk_array, className, tol, checklength)
% getPacedAltDoubleEvents  get non-error markers from EEG-markers of paced_alt_double trial
%
% USAGE: mrk_paced= getPacedAltDoubleEvents(mrk, mrk_array, className, tol, checklength)
%
% IN:  mrk           struct containing markers (pos, toe, fs)
%      mrk_array     array containing doubles [action_mrk, rest_mrk, pace_mrk, response_mrk, <click_mrk>]
%                    default for click_mrk: abs(response_mrk)
%      className     cell array containing class name strings for 1. click class, 2. no-click-class and 3. rest class.
%      tol           array with double entries for tolerance values.
%      checklength   multiple of 4 indicating the length of the array to check if there was no error in the typing rhythm. OPTIONAL.
%                    default: 40
%
% OUT: mrk_paced     struct containing class markers (pos, toe, fs, y, className, latency, indexedByEpochs)
%
% EXAMPLE:
% mrk       =
%    pos: [272, 3191, 3216, 3292, 3317, 3392, 3416, 3492, 3518, 3591]
%    toe: [25,  252,  12,   1,    12,   1,    11,   1,    11,   1]
%     fs: 100
% mrk_array = [10, 11, 1, -74, 74]
% className = {'right click', 'left no-click', 'rest'}
% tol       = [200, 500]



% 26.8.03 kraulem

if length(mrk_array)>4
  click_mrk = mrk_array(5);
else
  click_mrk = abs(mrk_array(4));
end

% provide classDef parameter for the call of makeABEevents:
classDef= { sprintf('S%d', mrk_array(3)), ...
             sprintf('not R%d(-%d,%d)',abs(mrk_array(4)), tol([2 2])), ...
             [mrk_array(1)], className{2}; 
            sprintf('S%d', mrk_array(3)), ...
             sprintf('R%d(-%d,%d)',abs(mrk_array(4)), tol([1 1])), ...
             [click_mrk], className{1};
%% the following is a temporary class: 
%% it will be removed before output of mrk_paced.
            sprintf('R%d', abs(mrk_array(4))), ...
             sprintf('R%d(-500,-1) and R%d(1,500)', abs(mrk_array([4 4]))), ...
             [9], 'erase error'};

if ~exist('checklength', 'var')
  checklength = 40;
end

for ind = 1:size(classDef,1)
  switch classDef{ind,4}
   case {'erase error'}
    err_ind = ind;
    err_mrk = classDef{ind,3};
   case {'left no-click', 'right no-click'}
    noclick_ind = ind;
    noclick_mrk = classDef{ind,3};
   case {'left click', 'right click'}
    click_ind = ind;
    click_mrk = classDef{ind,3};
   otherwise
    disp('unexpected fields in classDef found!');
  end
end


mrk_paced = makeABEevents(mrk, classDef);

% remove response_mrk and both neighbours, 
% plus all markers 4 secs before 'erase error' event:
eventarray = 1:length(mrk_paced.pos);
for ind1 = 1:length(mrk_paced.pos)
  if mrk_paced.toe(ind1) == err_mrk
    eventarray = eventarray(find( ...
        mrk_paced.pos(eventarray) <= mrk_paced.pos(ind1)-4*mrk_paced.fs | ...
        mrk_paced.pos(eventarray) >= mrk_paced.pos(ind1)+1*mrk_paced.fs ));
  end
end
mrk_paced.pos = mrk_paced.pos(eventarray);
mrk_paced.toe = mrk_paced.toe(eventarray);
%mrk_paced.y   = mrk_paced.y(sort([click_ind, noclick_ind]),eventarray);
%mrk_paced.className = mrk_paced.className(sort([click_ind, noclick_ind]));
%% eliminate the 'erase error' events
mrk_paced.y   = mrk_paced.y(:,eventarray);

% remove all runs which can't be trusted as double paced trials:
indexlow = min(find(mrk_paced.toe==click_mrk)); 
eventarray = [];
restarray = [];
eventindexarray = sort([1:4:checklength, 2:4:checklength]);
restindexarray  = sort([3:4:checklength, 4:4:checklength]);
while indexlow<=length(mrk_paced.pos)-checklength
  if IsDoublePaced(mrk_paced, click_mrk, indexlow, indexlow + checklength - 1)
    appendarray = indexlow:(indexlow + checklength - 1);
    eventarray = [eventarray, appendarray(eventindexarray)];
    restarray = [restarray,appendarray(restindexarray)];
    indexlow = indexlow + checklength;
  else % there is an error in this interval,
       % or the grid is not syncronized with the rhythm.
    indexlow = indexlow + 1;
  end
end

% detect rest class
mrk_paced.toe(restarray) = mrk_array(2);
mrk_paced.y(noclick_ind, restarray) = 0;
mrk_paced.y(err_ind,:) = zeros(1,length(mrk_paced.toe));
% the 'erase error' event is now overwritten by the 'rest' event
mrk_paced.y(err_ind,restarray) = 1;
mrk_paced.className(err_ind) = className(3);

% delete untrustworthy runs
keeparray = sort([eventarray, restarray]);
mrk_paced.pos = mrk_paced.pos(keeparray);
mrk_paced.toe = mrk_paced.toe(keeparray);
mrk_paced.y   = mrk_paced.y(:,keeparray);

% insert response jitter 
mrk_jit= getResponseJitter(mrk, {mrk_array(3), mrk_array(4)}, [tol(1) tol(1)]);

iSrc= find(ismember(mrk_jit.pos, mrk_paced.pos) & ~isnan(mrk_jit.latency));
iTrg= find(ismember(mrk_paced.pos, mrk_jit.pos(iSrc)));
mrk_paced.latency= NaN*ones(size(mrk_paced.pos));
mrk_paced.latency(iTrg)= mrk_jit.latency(iSrc);

% insert indexedByEpochs
mrk_paced.indexedByEpochs= {'latency'};

% sort classes
%mrk_paced= mrk_selectClasses(mrk_paced, [noclick_ind, click_ind, err_ind]);

return






function bool = IsDoublePaced(mrk, click_mrk, indexlow, indexup)
% decides if the array mrk.pos(indexlow:indexup) is a paced_alt_double-run, 
% starting with two "actions".
indexarr    = indexlow:indexup;
indexmodarr = 1:(indexup - indexlow+1);
% if there is a click in the "rest" period, it can't be a valid run.
bool = isempty(find(mrk.toe(indexarr(indexmodarr))==click_mrk & ...
                    ((indexmodarr-4*floor(indexmodarr/4))==3 | ...
                     (indexmodarr-4*floor(indexmodarr/4))==0)));

return
