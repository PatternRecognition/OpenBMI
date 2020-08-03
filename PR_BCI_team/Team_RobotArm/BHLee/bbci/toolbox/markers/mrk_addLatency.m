function mrk= mrk_addLatency(mrk, sgn, classes)
%mrk_addLatency(mrk, <sgn=1, classes>)
%
% adds (or subtracts for sgn=-1) the latency to the marker position.
% this might be used to convert stimulus-aligned to response-aligned
% markers (or vice versa).

% bb 09/03, ida.first.fhg.de

if ~exist('sgn', 'var') | isempty(sgn), sgn=1; end

ev= 1:length(mrk.pos);
if exist('classes', 'var'),
  clInd= getClassIndices(mrk, classes);
  ev= find(any(mrk.y(clInd,:), 1));
end

mrk.pos(ev)= mrk.pos(ev) + sgn*mrk.latency(ev);
