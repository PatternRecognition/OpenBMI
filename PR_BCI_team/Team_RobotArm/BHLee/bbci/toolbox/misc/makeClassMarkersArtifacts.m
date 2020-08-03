function mrk= makeClassMarkersArtifacts(mark)
%mrk= makeClassMarkers(mrk)
%
% IN   mrk             - event markers
%
% OUT  mrk       struct for class markers
%         .int       - intervals in which something happens
%         .artifacts      - names of the artifact
%         .y         - number of the artifacts which happens
%         .fs        - frequence


events = intersect(find(~strcmp(mark.toe,'stop')),find(~strcmp(mark.toe,'stopp')));
intervallength = ([events(2:end),length(mark.toe)+1]-events)>1;

mrk.artifacts = unique(cellstr(strvcat(mark.toe{events})))';

for i =1:length(events)
mrk.y(i) = find(strcmp(mark.toe{events(i)},mrk.artifacts));
end

mrk.int = [mark.pos(events); mark.pos(events+intervallength)]';

mrk.fs = mark.fs;
