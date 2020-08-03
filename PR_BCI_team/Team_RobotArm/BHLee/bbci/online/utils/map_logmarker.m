function log = map_logmarker(log,typ);


  
for i = 1:length(log);
  mrk = log(i).mrk;
  switch typ
   case 'response'
    ind = find(mrk.toe<0);
   case 'stimulus'
    ind = find(mrk.toe>0);
   otherwise
    error('type not known');
  end
  mrk.toe = abs(mrk.toe(ind));
  mrk.pos = mrk.pos(ind);
  log(i).mrk = mrk;
end
