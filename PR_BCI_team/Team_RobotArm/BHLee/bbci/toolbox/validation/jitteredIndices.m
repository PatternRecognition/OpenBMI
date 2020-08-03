function jidx= jitteredIndices(idx, jitter)
%jidx= jitteredIndices(idx, jitter)

jidx= idx;
for jj= jitter,
  jidx= [jidx idx+jj];
end
