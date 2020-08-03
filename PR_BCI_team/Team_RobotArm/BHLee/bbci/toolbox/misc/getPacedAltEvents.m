function mrk_paced = getPacedAltEvents(mrk, mrk_array, className, tol)
% getPacedAltEvents  get non-error markers from EEG-markers of paced_alt trial
%
% USAGE: mrk_paced= getPacedAltEvents(mrk, mrk_array, className, tol)
%
% IN:  mrk           struct containing markers (pos, toe, fs)
%      mrk_array     array containing doubles [action_mrk, rest_mrk, pace_mrk, response_mrk, <click_mrk>]
%                    default for click_mrk: abs(response_mrk)
%      className     cell array containing class name strings for 1. click class, 2. no-click-class and 3. rest class.
%      tol           array with double entries for tolerance values.
%
% OUT: mrk_paced     struct containing class markers (pos, toe, fs, y, className, indexedByEpochs)
%
% EXAMPLE:
% mrk =
%    pos: [272, 3191, 3216, 3292, 3317, 3392, 3416, 3492, 3518, 3591]
%    toe: [25,  252,  12,   1,    12,   1,    11,   1,    11,   1]
%     fs: 100
% mrk_array = [10, 11, 1, -74, 74]
% className = {'right click', 'left no-click', 'rest'}
% tol       = [200, 500]


% 22.8.03 kraulem

%classDef = { 'S1', 'not R74(-500,500)', [10], 'left no-click';
%           'R74', 'R74(-500,-1) and R74(1,500)', [9], 'erase error';% temporary class: will be removed before output of mrk_paced.
%           'S1', 'R74(-200,200)', [74], 'right-click'};


if length(mrk_array)>4
  click_mrk = mrk_array(5);
else
  click_mrk = abs(mrk_array(4));
end

% provide classDef parameter for the call of makeABEevents:
classDef= { sprintf('S%d', mrk_array(3)), ...
             sprintf('not R%d(-%d,%d)', abs(mrk_array(4)), tol([2 2])), ...
             [mrk_array(1)], className{2};
            sprintf('R%d', abs(mrk_array(4))), ...
             sprintf('R%d(-500,-1) and R%d(1,500)', abs(mrk_array([4 4]))), ...
             [9], 'erase error'; %% temporary class:
                                 %% will be removed before output of mrk_paced.
            sprintf('S%d', mrk_array(3)), ...
             sprintf('R%d(-%d,%d)', abs(mrk_array(4)), tol([1 1])), ...
             [click_mrk], className{1}};

mrk_paced = makeABEevents(mrk, classDef);

%% remove response_mrk and both neighbours, 
%% plus all markers 4 secs before 'erase error' event:
eventarray = 1:length(mrk_paced.pos);
for ind1 = 1:length(mrk_paced.pos)
  if mrk_paced.toe(ind1) == 9
    eventarray = eventarray(find( ...
        mrk_paced.pos(eventarray) <= mrk_paced.pos(ind1)-4*mrk_paced.fs | ...
        mrk_paced.pos(eventarray) >= mrk_paced.pos(ind1)+1*mrk_paced.fs ));
  end
end
mrk_paced.pos = mrk_paced.pos(eventarray);
mrk_paced.toe = mrk_paced.toe(eventarray);
mrk_paced.y   = mrk_paced.y([1,3],eventarray);
mrk_paced.className = mrk_paced.className([1,3]);
% eliminate the 'erase error' events

% remove all click_mrk with uneven number of s11 between them
lowerindex = min(find(mrk_paced.toe==click_mrk));
eventarray = [lowerindex];
restarray = [];
appendarray = [];
for ind1 = (lowerindex+1):length(mrk_paced.pos)
  appendarray = [appendarray, ind1];
  if mrk_paced.toe(ind1)==click_mrk
    if gcd(length(appendarray),2)==2
      eventarray = [eventarray,appendarray];
      restarray = [restarray,appendarray(1:2:length(appendarray-1))];
       %% only every second event is presumed to be a 'left no-click',
       %% the other events are 'rest'
    end
    appendarray = [];
  end
end
mrk_paced.toe(restarray) = mrk_array(2);  %% detect rest class
mrk_paced.y(1,restarray) = 0;
mrk_paced.y(3,:) = zeros(1,length(mrk_paced.toe));
mrk_paced.y(3,restarray) = 1;
mrk_paced.className(3) = className(3);

mrk_paced.pos = mrk_paced.pos(eventarray);% delete uneven-number-runs
mrk_paced.toe = mrk_paced.toe(eventarray);
mrk_paced.y   = mrk_paced.y(:,eventarray);

% insert response jitter 
mrk_jit= getResponseJitter(mrk, {mrk_array(3), mrk_array(4)}, [tol(1) tol(1)]);

iSrc= find(ismember(mrk_jit.pos, mrk_paced.pos) & ~isnan(mrk_jit.latency));
iTrg= find(ismember(mrk_paced.pos, mrk_jit.pos(iSrc)));
mrk_paced.latency= NaN*ones(size(mrk_paced.pos));
mrk_paced.latency(iTrg)= mrk_jit.latency(iSrc);

% insert indexedByEpochs
mrk_paced.indexedByEpochs= {'latency'};

return
