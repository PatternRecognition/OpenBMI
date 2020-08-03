labels={'4 Hz','5 Hz','6 Hz','7.5 Hz','8 Hz','10 Hz','12 Hz','15 Hz'};
freq_array=[4 5 6 7.5 8 10 12 15];
stimDef= {1, 2, 3, 4, 5, 6, 7, 8, [100 249 253]; labels{:}, 'stop'};
mrk= mrk_defineClasses(mrk_orig, stimDef);
%mrk=mrk_selectClasses(mrk,[2 3 4 5 6 8 9]);

blk= [];
for ff= 1:size(mrk.y,1)-1,
  cn=char(mrk.className{ff});
  blk0= blk_segmentsFromMarkers(mrk, ...
    'start_marker',cn, ...
    'end_marker','stop');
  blk0.className= {cn};
  blk= blk_merge(blk, blk0);
end
[Cnt0, blkcnt]= proc_concatBlocks(cnt0, blk);





%dat.x=randn(6,63)
clear output
%define packet size:
ps=4;
tic
iter=6000;
Cnt1=proc_selectChannels(Cnt0,'not','F*','T*','C*','Ref');

for i=1:iter
dat.x=Cnt1.x((i-1)*ps+1:(i)*ps,:);
%dat.x
dat.clab=Cnt1.clab;

if i==1
[dat,state]= online_filterbank(dat, [], analyze.csp_b,analyze.csp_a);
else
  [dat,state]= online_filterbank(dat,state, analyze.csp_b,analyze.csp_a);
end
%dat.x=reshape(dat.x,ps,21,16);
out= proc_linearDerivationSSVEP(dat, frequency_matrix,analyze.csp_w);
out=proc_variance(out);
out=proc_logarithm(out);
size(out.x);
out=apply_separatingHyperplaneSSVEP(cls.C,squeeze(out.x));
output(i,:)=out;
1/toc;
tic;
end
%output=output-repmat(mean(output),iter,1);
figure;plot(ps*[1:iter],output)
