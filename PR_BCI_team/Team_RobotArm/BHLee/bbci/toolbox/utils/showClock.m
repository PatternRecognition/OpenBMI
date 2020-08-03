function showClock(r, rounds, everySecs)
%showClock(r, rounds, everySecs)

if nargin<3 everySecs=1; end

global lastShow
cloc= clock;
if isempty(lastShow) | any(cloc(1:2)>lastShow(1:2)), 
  lastShow= clock; 
  lastShow(6)= lastShow(6)-everySecs; 
end

if r==rounds
  h= pie(1, {'finished'}); 
  set(h(1), 'faceColor', 'g'); set(gca,'xDir','reverse')
  drawnow;
elseif etime(clock, lastShow)>everySecs
  clf
  h= pie([r rounds-r],{['done: ' int2str(r)],['to do: ' int2str(rounds-r)]}); 
  set(h(1), 'faceColor', 'g'); set(h(3), 'faceColor', 'r'); 
  set(gca,'xDir','reverse'); 
  drawnow;
  lastShow= clock;
end
