function err_out= mark_errors(comb_out, goal, err_out, fs)
%err_out= mark_errors(comb_out, goal, err_out, fs)
%
% error types:
%  1: false negative (no classification in [-500 0] to event)
%  2: false left/right classification
%  3: false event detection

persistent lastUpto iClStart

if strcmp(comb_out, 'init'),
  lastUpto= [];
  err_out= NaN*ones(size(goal,1), 3);
  return;
end

preLimit= 1.2;
postLimit= 0.3;
lastGrace= round(0.1*fs);
halfSec= round(fs/2);

upto= max(find(abs(goal)==1));
if isempty(lastUpto) | upto<lastUpto,
  lastUpto= 0;
end
if isempty(upto) | upto==lastUpto, return; end

iv= lastUpto+1:upto; 
for n= iv,
  type= 0;
  lastHalf= max(n-halfSec,1):n;
  if goal(n) & ~any(comb_out(lastHalf)),
    type= 1;
  else
    iLastEvent= max(find(goal(1:n)));
    iNextEvent= n+min(find(goal(n+1:end)));
    cl= sign(comb_out(n));
    if cl,
      if n==1 | cl~=comb_out(n-1),
        iClStart= n;
      end

      %% refer to previous event
      if ~isempty(iLastEvent) & ...
            (iClStart-iLastEvent)/fs<=postLimit & ...
            iClStart<=iLastEvent+lastGrace,
        if cl~=goal(iLastEvent),
          type= 2;
        end
        
      %% refer to upcoming event  
      elseif ~isempty(iNextEvent) & ...
            (iNextEvent-n)/fs<=preLimit,
        if cl~=goal(iNextEvent),
          type= 2;
        end
        
      %% no event to refer to
      else
        type= 3;
      end
    end
  end
  if type,
    err_out(n,type)= 1;
  end
end    
lastUpto= upto;
err_out(iv,1)= err_out(iv,1).*goal(iv);
err_out(iv,2:3)= err_out(iv,2:3).*repmat(sign(comb_out(iv)),1,2);
