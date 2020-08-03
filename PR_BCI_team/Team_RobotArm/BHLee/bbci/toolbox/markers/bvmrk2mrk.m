function mrk= bvmrk2mrk(bvmrk, fs)

mrk.pos= [bvmrk.pos];
mrk.type= {bvmrk.type};
mrk.desc= {bvmrk.desc};
mrk.length= [bvmrk.length];
mrk.chan= [bvmrk.chan];
mrk.time= {bvmrk.time};
mrk.indexedByEpochs= {'type','desc','length','time','chan'};

if nargin>1,
  mrk.fs= fs;
end
