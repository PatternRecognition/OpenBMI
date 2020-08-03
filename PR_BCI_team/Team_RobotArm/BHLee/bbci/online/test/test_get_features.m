feature.cnt = 1;
feature.ilen_apply = 1000;
feature.proc = {'proc_linearDerivation','proc_jumpingMeans'};
feature.proc_param = {{[1;1]}, {100}};
opt = struct('fs',100);
cont_proc = struct('clab',{{'1','2'}});


storeContData('init',1,2,'ringBufferSize',30000,'fs',100);

getFeature('init',feature,opt,cont_proc);

pos = 0;


for i = 1:100
  pos = pos+randn;
  storeContData('append',1,[pos+rand*0.1 pos+rand*0.1; pos+rand*0.1 pos+rand*0.1;pos+rand*0.1 pos+rand*0.1;pos+rand*0.1 pos+rand*0.1]);
end

tic
fv = getFeature('apply',1,-20);
toc

tic
fv = getFeature('apply',1,-20);
toc

getFeature('reset');
tic
fv = getFeature('apply',1,-20);
toc





