file_dir= [EEG_IMPORT_DIR 'bci_competition_iii/martigny/'];


for su=1:3;

file_list= strcat(file_dir, 'train_subject', int2str(su), '_raw0', ...
                  cellstr(int2str([1:3]')));
for ff= 1:length(file_list),
  S= load(file_list{ff});
  if ff==1,
    cnt= struct('x', S.X, 'clab',{S.nfo.clab}, 'fs',S.nfo.fs, ...
                'title', untex(S.nfo.name(1:end-2)));
    Y= S.Y;
  else
    cnt.x= cat(1, cnt.x, S.X);
    Y= cat(1, Y, S.Y);
  end
  cnt.T(ff)= size(S.X, 1);
end
mnt= setDisplayMontage(S.nfo.clab, 'martigny');
mnt.xpos= S.nfo.xpos;
mnt.ypos= S.nfo.ypos;
clear S

mrk= struct('fs',cnt.fs);
mrk.pos= find(diff([0; Y]))';
break_points= cumsum(cnt.T(1:end-1));
mrk.pos= unique([mrk.pos break_points+1]);
mrk.y= double([Y(mrk.pos)'==2; Y(mrk.pos)'==3; Y(mrk.pos)'==7]);
mrk.className= {'left', 'right', 'word'};
mrk.toe = transpose(cellstr(num2str(([1 2 3]*mrk.y)')));
mrk.block = ones(1,sum(mrk.pos<break_points(1)+1));

for i = 1:length(break_points)-1
  mrk.block = cat(2,mrk.block,(i+1)*ones(1,sum(mrk.pos>=break_points(i)+1 & mrk.pos<break_points(i+1)+1)));
end
mrk.block =  cat(2,mrk.block,(length(break_points)+1)*ones(1,sum(mrk.pos>=break_points(end)+1)));
mrk.indexedByEpochs = {'block'};


model_RLDA= struct('classy', 'RLDA');
model_RLDA.param= [0 0.01 0.1 0.3 0.5 0.7];
model_LDA= 'LDA';


opt_xv= struct('out_trainloss',1, 'outer_ms',0, 'xTrials',[1 1],'msTrials',[1 1],...
               'verbosity',2);
[b,a] = butter(5,[8 15]/cnt.fs*2);
cnt_flt = proc_laplace(cnt,'diagonal','','filter all');
cnt_flt = proc_filt(cnt_flt,b,a);
epo = makeEpochs(cnt_flt,mrk,[0 min_len]);

te_long = zeros(4,2);
st_long = zeros(4,2);
te_long_stat = zeros(4,2);
st_long_stat = zeros(4,2);

epo = proc_variance(epo);
epo = proc_logarithm(epo);
for i = 1:4
  switch i
   case 1
    fv= proc_selectClasses(epo, 'left','right');
   case 2
    fv= proc_selectClasses(epo, 'left','word');
   case 3
    fv= proc_selectClasses(epo, 'right','word');
   case 4
    fv = epo;
  end
  [te_long(i,:),st_long(i,:)] = xvalidation(fv, model_RLDA, opt_xv);
  opt2 = opt_xv;
  opt2 = rmfield(opt2,'xTrials');
  
  opt2.divTr = {{find(fv.block<max(fv.block))}};
  opt2.divTe = {{find(fv.block==max(fv.block))}};
  [te_long_stat(i,:),st_long_stat(i,:)] = xvalidation(fv, model_RLDA, opt2);
end

mrk.end = size(cnt.x,1);
mrk_sh = mrk_setMarkers(mrk,[1000,0,1000],1);

epo = makeEpochs(cnt_flt,mrk_sh,[-1000 0]);
te_short = zeros(4,2);
st_short = zeros(4,2);
te_short_stat = zeros(4,2);
st_short_stat = zeros(4,2);

epo = proc_variance(epo);
epo = proc_logarithm(epo);
for i = 1:4
  switch i
   case 1
    fv= proc_selectClasses(epo, 'left','right');
   case 2
    fv= proc_selectClasses(epo, 'left','word');
   case 3
    fv= proc_selectClasses(epo, 'right','word');
   case 4
    fv = epo;
  end
  [te_short(i,:),st_short(i,:)] = xvalidation(fv, model_RLDA, opt_xv);
  opt2 = opt_xv;
  opt2 = rmfield(opt2,'xTrials');
  ind = [1,find(diff(fv.bidx)~=0)+1];
  opt2.divTr = {{fv.bidx(ind(find(fv.block(ind)<max(fv.block))))}};
  opt2.divTe = {{fv.bidx(ind(find(fv.block(ind)==max(fv.block))))}};
  [te_short_stat(i,:),st_short_stat(i,:)] = xvalidation(fv, model_RLDA, opt2);
end


mrk_train = mrk_selectEvents(mrk,find(mrk.block<max(mrk.block)));
epo = makeEpochs(cnt_flt,mrk_train,[0 min_len]);

epo = proc_variance(epo);
epo = proc_logarithm(epo);
out = cell(1,4);

for i = 1:4
  switch i
   case 1
    fv= proc_selectClasses(epo, 'left','right');
   case 2
    fv= proc_selectClasses(epo, 'left','word');
   case 3
    fv= proc_selectClasses(epo, 'right','word');
   case 4
    fv = epo;
  end
  opt2 = opt_xv;
  opt2.train_only = 1;
  opt2 = rmfield(opt2,'xTrials');
  opt2.divTr = {{1:size(fv.x,3)}};
  opt2.divTe = {{[]}};
  C = xvalidation(fv, model_RLDA, opt2);
  
  ep = copyStruct(cnt_flt,'x');
  poi = 1;
  ou = [];
  for po = sum(cnt_flt.T)+1000:62.5:size(cnt_flt.x,1)*1000/512
    ep.x = cnt_flt.x(round((po-1000)*cnt_flt.fs/1000+1:po*cnt_flt.fs/1000),:);
    ep = proc_variance(ep);
    ep = proc_logarithm(ep);
    ou = cat(2,ou,apply_separatingHyperplane(C,ep.x(:)));
  end
  
  out{i} = ou;
  
  
end

po = sum(cnt_flt.T)+1000:62.5:size(cnt_flt.x,1)*1000/512;
lab = zeros(1,length(po));
for i = 1:length(po);
  ind = max(find(mrk.pos*1000/512<=po(i)));
  if po(i)-mrk.pos(ind)*1000/512 <=1000
    lab(i) = 0;
  else
    lab(i) = [1,2,3]*mrk.y(:,ind);
  end
end

lab = repmat(lab,[4,1]);
lab(1,find(lab(1,:)==3))=0;
lab(2,find(lab(2,:)==2))=0;
lab(3,find(lab(3,:)==1))=0;

ou = zeros(4,length(po));
ou(1,:) = 0.5*sign(out{1})+1.5;
ou(2,:) = sign(out{2})+2;
ou(3,:) = 0.5*sign(out{3})+2.5;
[dum,ou(4,:)] = max(out{4},[],1);

res = sum((ou==lab),2)./sum(lab>0,2);

fid = fopen([TEX_DIR 'bci_competition_iii/martigny/results_subject_' int2str(su) '_spectral_feature.tex'],'w');

fprintf(fid,'left-right & %2.1f & %2.1f & %2.1f & %2.1f & %2.1f \\\\\n',te_long(1,1)*100,te_long_stat(1,1)*100,te_short(1,1)*100,te_short_stat(1,1)*100,100-100*res(1));
fprintf(fid,'left-word & %2.1f & %2.1f & %2.1f & %2.1f & %2.1f \\\\\n',te_long(2,1)*100,te_long_stat(2,1)*100,te_short(2,1)*100,te_short_stat(2,1)*100,100-100*res(2));
fprintf(fid,'right-word & %2.1f & %2.1f & %2.1f & %2.1f & %2.1f \\\\\n',te_long(3,1)*100,te_long_stat(3,1)*100,te_short(3,1)*100,te_short_stat(3,1)*100,100-100*res(3));
fprintf(fid,'all & %2.1f & %2.1f & %2.1f & %2.1f & %2.1f\n',te_long(4,1)*100,te_long_stat(4,1)*100,te_short(4,1)*100,te_short_stat(4,1)*100,100-100*res(4));

fclose(fid);


 



end
