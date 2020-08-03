function mrkout= mrk_arrayOfStructs2structOfArrays(mrk, fs)

mrkout.pos= [mrk.pos];
mrkout.type= {mrk.type};
mrkout.desc= {mrk.desc};
if nargin>1,
  mrkout.fs= fs;
end
mrkout.length= [mrk.length];
mrkout.chan= [mrk.chan];
mrkout.time= {mrk.time};
mrkout.indexedByEpochs= {'type','desc','length','time','chan'};
