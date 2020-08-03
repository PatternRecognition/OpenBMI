function mrk= getPaced_cncrEvents(file, marker_stimuli, classNames, tol, ...
                                   blockingDoubles, varargin)
%mrk= getPaced_cncrEvents(file, marker, classNames, tol=[200 700], ...
%                         <blockingDoubles=55, fs=100>)
%
% marker= [action_marker, rest_marker], e.g., [11 12]
%   or
% marker= [action_marker, rest_marker, new_click_marker], e.g. [11 12 10]
%   where 10 is the new marker that will be assigned to the click class.
% tol= [click_tolerance, no-click_tolerance]
% classNames, e.g., {'left click', 'right no-click', 'rest'}

if ~exist('tol', 'var') | isempty(tol), tol= [200 700]; end
if ~exist('blockingDoubles', 'var') | isempty(blockingDoubles), 
  blockingDoubles=55; 
end

if length(marker_stimuli)>3
  marker_pace = marker_stimuli(4);
else
  marker_pace= 1;  %% beat of pace maker is marked 1 (BV marker S  1)
end


Mrk= readMarkerTable(file, varargin{:});

iPace= find(Mrk.toe==marker_pace);
iStimulus= find(ismember(Mrk.toe, marker_stimuli));
iAllKeys= find(ismember(Mrk.toe, -['A':'Z', 192]));

nEvents= length(iStimulus);
mrk.pos= zeros(1, nEvents);
mrk.toe= zeros(1, nEvents);
mrk.y= zeros(3, nEvents);
mrk.latency= NaN*zeros(1, nEvents);
mrk.indexedByEpochs= {'latency'};
reject= [];
ip= 0;
ep= length(iPace);
for ei= 1:nEvents,
  is= iStimulus(ei);

  if ip==ep,
    warning('missing pace marker after stimulus');
    break;
  end
  pp= min( find(Mrk.pos(iPace(ip+1:ep)) > Mrk.pos(is)) );
  ip= ip+pp;
  mrk.pos(ei)= Mrk.pos(iPace(ip));

  if Mrk.toe(is)==marker_stimuli(1),
    [absjit,ik]= min(abs( Mrk.pos(iAllKeys) - mrk.pos(ei) ));
    absjitMsec=  absjit/Mrk.fs*1000;
    if absjitMsec < tol(1),
      if ik<length(iAllKeys),
        diff= (Mrk.pos(iAllKeys(ik+1)) - Mrk.pos(iAllKeys(ik))) / Mrk.fs*1000;
        if diff<blockingDoubles,
          reject= [reject, ei];
        end
      end
      iKey= iAllKeys(ik);
      if length(marker_stimuli)<3,
        mrk.toe(ei)= -Mrk.toe(iKey);
      else
        mrk.toe(ei)= marker_stimuli(3);
      end
      mrk.y(1,ei)= 1;
      mrk.latency(ei)= Mrk.pos(iKey) - mrk.pos(ei);
    elseif absjitMsec > tol(2),
      mrk.toe(ei)= Mrk.toe(is);
      mrk.y(2,ei)= 1;
    else
      warning(sprintf('ambiguous response at %d s encountered: rejected', ...
                      round(Mrk.pos(iKey)/Mrk.fs)));
    end
  else
    [absjit,ik]= min(abs( Mrk.pos(iAllKeys) - mrk.pos(ei) ));
    absjitMsec=  absjit/Mrk.fs*1000;
    if absjitMsec < tol(1),
      warning('click after rest stimulus: rejected');
    else
      mrk.toe(ei)= Mrk.toe(is);
      mrk.y(3,ei)= 1;
    end
  end
end

accept= find(any(mrk.y,1));
mrk.toe= mrk.toe(accept);
mrk.pos= mrk.pos(accept);
mrk.latency= mrk.latency(accept);
mrk.y= mrk.y(:,accept);
mrk.fs= Mrk.fs;

if ~isempty(classNames),
  mrk.className= classNames;
end
