function mrk = separate_predecessor(mrk,distance)
%DIVIDE_FOR_PREVIOUS_EVENT divides each label in number of classes
%labels regarding the previous event
%
% usage:
%     mrk = divide_for_previous_event(mrk,<distance>);
%
% input:
%     mrk      a usual mrk structure
%     distance all labels which have no previous element in this time
%              (regarding the mrk.pos structure) are omitted. Default:
%              all labels are taken except the first one.
%
% output
%     mrk      a updated mrk structure
%
% GUIDO DORNHEGE, 02/04/03

nMarkers = length(mrk.pos);
if ~exist('distance','var') | isempty(distance)
  ind = 2:nMarkers;
else
  ind = find(mrk.pos(2:end)-mrk.pos(1:end-1)<=distance)+1;
end

nClasses = length(mrk.className);
className = cell(1,nClasses^2);
labels = zeros(nClasses^2,length(ind));

for i = 1:length(mrk.className)
  event2 = mrk.className{i};
  ind2 = find(mrk.y(i,ind));  
  for j = 1:length(mrk.className)
    event1 = mrk.className{j};
    k = (i-1)*nClasses+j;
    className{k} = sprintf('%s/%s',event1,event2);
    labels(k,ind2(find(mrk.y(j,ind(ind2)-1))))=1;
  end
end

mrk.pos = mrk.pos(ind);
mrk.toe = mrk.toe(ind);
mrk.className = className;
mrk.y = labels;