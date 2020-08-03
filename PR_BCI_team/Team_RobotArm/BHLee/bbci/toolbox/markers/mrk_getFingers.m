function mrk= mrk_getFingers(mrk, fingerMarkers, fingerLabels)
%mrk= mrk_getFingers(mrk, <fingerMarkers, fingerLabels>)
%
% input:
%    mrk             a usual mrk structure for selfpaced
%    fingerMarkers   cell array containing the markers that are used
%                    in mrk.toe to specify the fingers, default
%                    {'A','S','D',['F' 1],['J' 2],'K','L',192}
%    fingerLabels    cell array containing the corresponding class names,
%                    default {'left V', 'left IV', ..., 'right V'}
%
% output:
%    mrk             a mrk structure where each finger is a class

% Guido Dornhege, Volker Kunzmann 04/07/2003
% modified bb

if ~exist('fingerMarkers','var'),
  fingerMarkers= {'A','S','D',['F' 1],['J' 2],'K','L',192};
end
if ~exist('fingerLabels','var'),
  fingerLabels= {'left V', 'left IV', 'left III', 'left II', ...
                 'right II', 'right III', 'right IV', 'right V'};
end

fi = unique(abs(mrk.toe));
mrk.y = zeros(length(fi),size(mrk.y,2));
mrk.className = cell(1,length(fi));

for i = 1:length(fi)
  j= 1;
  while j<=length(fingerMarkers) & ~ismember(fi(i), fingerMarkers{j}),
    j= j+1;
  end
  if j>length(fingerMarkers),
    error(sprintf('no match for marker type %d found.', fi(i)));
  end
  ind = find(abs(mrk.toe)==fi(i));
  mrk.y(i,ind) = 1;
  mrk.className{i} = fingerLabels{j};
end
