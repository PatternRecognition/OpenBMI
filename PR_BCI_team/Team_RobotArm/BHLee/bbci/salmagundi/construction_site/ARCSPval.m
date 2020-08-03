file = {'Ben_02_07_23/imagmultimodalBen','Thorsten_02_07_31/imagmultimodalThorsten','Steven_02_08_12/imagmultiSteven','Hendrik_02_08_12/imagmultiHendrik'};

name = {'Benjaminmultimodal','Thorstenmultimodal','Steven','Hendrik'};

feature = {{{'leg','auditory'},{'leg','visual'},{'auditory','visual'}},{{'leg','auditory'},{'leg','visual'},{'auditory','visual'}},{{'left','right'},{'left','foot'},{'left','auditory'},{'left','visual'},{'left','tactile'},{'right','foot'},{'right','auditory'},{'right','visual'},{'right','tactile'},{'foot','auditory'},{'foot','visual'},{'foot','tactile'},{'auditory','visual'},{'auditory','tactile'},{'visual','tactile'}},{{'left','right'},{'left','foot'},{'left','auditory'},{'left','visual'},{'left','tactile'},{'right','foot'},{'right','auditory'},{'right','visual'},{'right','tactile'},{'foot','auditory'},{'foot','visual'},{'foot','tactile'},{'auditory','visual'},{'auditory','tactile'},{'visual','tactile'}}};



model.classy = 'RLDA';
model.param.index = 2;
model.param.scale = 'lin';
model.param.value = [0,0.0005,0.002,0.005,0.01,0.05,0.1,0.2,0.4,0.6,0.8,1];
model.msDepth = 3;

ARCOEF = {2,3,4,5,6,8,10};
for i = 1:length(file)
     fi = file{i};
     feat = feature{i};
     for j = 1:length(feat)
     fe = feat{j};
     [cnt,mrk,mnt] = loadProcessedEEG(fi);
     ind = zeros(1,2);
     ind(1) = find(strcmp(mrk.className,fe{1}));
     ind(2) = find(strcmp(mrk.className,fe{2}));

     mrk2 = mrk;
     mrk2.className = fe;
     mrk2.y = mrk.y(ind,:);
     pla = find(sum(mrk2.y));
     mrk2.y = mrk2.y(:,pla);
     mrk2.pos = mrk2.pos(pla);
     mrk2.toe = mrk2.toe(pla);

     
     load(['/home/tensor/dornhege/Combination/bestResults/BEST' name{i} fe{1} fe{2} 'CSP']);

     
     cnt = proc_filtForth(cnt,frequencies);
     epo = makeSegments(cnt,mrk2,[-1000 3500]);
     clear cnt
clear mrk
clear mnt
     if sum(baseline~=0)>0
     epo = proc_baseline(epo,baseline);
end
if laplacefil 
epo = proc_laplace(epo,'small');
end
epo = proc_selectChannels(epo,Channel{:});
epo = proc_selectIval(epo,intervall{1});

epo.proc = ['fv = proc_csp(epo,' num2str(COEFS) '); fv = proc_variance(fv); fv = proc_logNormalize(fv);'];

     fprintf('\nName: %s, Feature: %s-%s, only CSP\n\n',name{i},fe{1},fe{2});

     modell = selectModel(epo,model,[3 10 round(0.9*size(epo.y,2))]);
     [te,tr,out,avE,evE] = doXvalidationPlus(epo,modell,[10 10]);
     [func,param] = getFuncParam(modell);
if abs((te(1)-test)/test)>0.1
     error('ABweichung');
end
     save(['/home/tensor/dornhege/ARCSP/ergs/' name{i} fe{1} fe{2} '0'],'te','tr','out','avE','evE','param','frequencies','baseline','laplacefil','Channel','intervall','COEFS');

for k = 1:length(ARCOEF)
     arco = ARCOEF{k};
     epo.proc = ['fv = proc_csp(epo,' num2str(COEFS) '); fv = proc_arCoefsPlusVar(fv,' num2str(arco) ');'];
     %epo.proc = ['fv = proc_csp(epo,' num2str(COEFS) '); fv = proc_arCoefsPluslogNormVar(fv,' num2str(arco) ');'];
     fprintf('\nName: %s, Feature: %s-%s, AR: %i\n\n',name{i},fe{1},fe{2},arco);

     modell = selectModel(epo,model,[3 10 round(0.9*size(epo.y,2))]);
     [te,tr,out,avE,evE] = doXvalidationPlus(epo,modell,[10 10]);
     [func,param] = getFuncParam(modell);
     
     save(['/home/tensor/dornhege/ARCSP/ergs/' name{i} fe{1} fe{2} num2str(arco)],'te','tr','out','avE','evE','param','arco','frequencies','baseline','laplacefil','Channel','intervall','COEFS');
end
end
end
