state= acquire_bv(100, 'localhost');
while 1,
  [dmy,bn,mp,md,mt]= acquire_bv(state);
  for mm= 1:length(mt),
    %nowstr= datestr(now,31);
    %fprintf('marker %s at %s\n', md{mm}, nowstr);
    fprintf('marker %s at %d\n', md{mm}, bn+mp(mm));
  end
end
acquire_bv('close');
