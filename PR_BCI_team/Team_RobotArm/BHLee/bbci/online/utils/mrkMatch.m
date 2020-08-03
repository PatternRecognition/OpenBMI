function index = mrkMatch(mrkFound,mrkArray)
% match a marker to a cell array of marker or marker templates.
%
% Matthias Krauledat, 03/12/2004
% TODO: extended documentation by Schwaighase
% $Id: mrkMatch.m,v 1.1 2006/04/27 14:24:59 neuro_cvs Exp $

index = [];
mrkStr = dec2bin(mrkFound);
for i = 1:length(mrkArray)
  if isnumeric(mrkArray{i})
    % mrkArray{i} contains a double, indicating the marker.
    if mrkArray{i}==mrkFound
      index = i;
      break;
    end
  else
    % mrkArray{i} contains a string, possibly with wildcards.
    if length(mrkArray{i})<length(mrkStr)
      % this can't be the correct string; too few wildcards.
      continue;
    elseif any(mrkArray{i}(1:(length(mrkArray{i})-length(mrkStr)))~='*')
      % this can't be the correct string; too many trailing non-wildcards.
      continue;
    elseif all(mrkArray{i}(find(mrkArray{i}((length(mrkArray{i})- ...
					     length(mrkStr)+1):end)~= ...
				mrkStr)+(length(mrkArray{i})-length(mrkStr)))=='*')
      % just compare over the length of str. If any character doesn't match and
      % it is a wildcard, then the strings match.
      index = i;
    else
      % this can't be the correct string: conflicting digits.
      continue;
    end
  end
end
return