function mrk = mrkFromTriggerChannel(cnt,varargin)
% mrk = mrkFromTrigger(cnt,opt)
%
% extract marker structure from a trigger channel in the cnt struct. 
%
% IN:  cnt - continuous data structure, containing a channel with trigger
%            signals.
%      opt - options struct or parameter list; possible fields:
%        .channel  - name of the channel with the trigger signal.
%                    default: 'Trigger'
%        .steps    - double array with the potential levels that are to be 
%                    translated into markers.
%                    default: linspace(-7,7,256)
%        .trg_mrk  - the markers into which the above steps are 
%                    to be translated. 0 means "no marker".
%                    default: 0:255
%        .min_len  - minimal length of a markerperiod in ms
%                    default: 20
%        .block_same_toe - if this is true, a new marker of the same 
%                    type can only start after 0 or a different 
%                    marker.
%                    default: true.
%        .blocking_time - the minimal time after which a new marker is set.
% OUT: mrk - BV-format marker struct.
%
%
% TODO: make this function more reliable against noise. Right now, the 
%       triggers of nearby voltage levels are likely to be confused.

% kraulem 10/06

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
		  'channel','Trigger',...
		  'steps',linspace(-7,7,256),...
		  'trg_mrk',0:255,...
		  'min_len',20,...
		  'block_same_toe',1,...
                  'blocking_time',0);

% sort the input steps:
[opt.steps,sort_ind] = sort(opt.steps);
opt.trg_mrk = opt.trg_mrk(sort_ind);

trigChan = chanind(cnt.clab,opt.channel);
last_mrk = -inf;
last_toe = 0;

% prepare the marker struct:
mrk = [];
mrk.type = {};
mrk.desc = {};
mrk.pos = [];
mrk.length = [];
mrk.chan = [];
mrk.time = {};
mrk.fs = cnt.fs;
mrk.indexedByEpochs = {'type'  'desc'  'length'  'time'  'chan'};

for ii = 1:(size(cnt.x,1)-opt.min_len/1000*cnt.fs)
  % check if the sample is too close to the old marker:
  if (ii-last_mrk)/cnt.fs*1000>max(opt.min_len,opt.blocking_time)
    %check if it could be a start of a new markerperiod:
    [d,ind1] = min(abs(cnt.x(ii,trigChan)-opt.steps));
    if opt.trg_mrk(ind1)~=0
      % only nonzero markers count.
      av = mean(cnt.x(ii:floor(ii+opt.min_len/1000*cnt.fs),trigChan));
      [d,ind2] = min(abs(av-opt.steps));
      if ind1==ind2 & (~opt.block_same_toe | opt.trg_mrk(ind1)~=last_toe)
	% this is a new marker.
	mrk.type{end+1} = 'Stimulus';
	mrk.desc{end+1} = ['S' num2str(opt.trg_mrk(ind1))];
	mrk.pos(end+1) = ii;
	mrk.length(end+1) = 1;
	mrk.chan(end+1) = 0;
	mrk.time{end+1} = '';
	last_toe = opt.trg_mrk(ind1);
        last_mrk = ii;
      end
    else
      % the trigger is at zero, so the last marker interval is over.
      last_mrk = -inf;
      last_toe = 0;
    end
  end
end