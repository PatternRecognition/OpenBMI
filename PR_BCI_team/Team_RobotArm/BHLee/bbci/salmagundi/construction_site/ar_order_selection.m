file_list= {'Thorsten_02_07_31/imagmultimodalThorsten', {'leg','auditory'}; ...
            'Steven_02_08_12/imagmultiSteven', {'left','right'}; ...
            'Steven_02_08_12/imagmultiSteven', {'left','auditory'}; ...
            'Steven_02_08_12/imagmultiSteven', {'left','tactile'}; ...
            'Soeren_02_08_13/imagmultiSoeren', {'left','right'}; ...
            'Soeren_02_08_13/imagmultiSoeren', {'left','foot'}; ...
            'Soeren_02_08_13/imagmultiSoeren', {'left','tactile'}};

proc_list= {[7 14], [ 500 3000], 4,  [ 500 3000], 5; ...
            [7 14], [ 500 2500], 4,  [ 500 3000], 5; ...
            [7 14], [ 500 3500], 4,  [ 500 3500], 5; ...
            [7 30], [ 500 3500], 4,  [ 500 2500], 5; ...
            [7 14], [1000 2000], 2,  [ 500 2000], 5; ...
            [7 14], [ 500 2500], 4,  [ 500 2500], 6; ...
            [7 14], [ 500 2000], 2,  [ 500 2000], 6};

model.classy= 'RLDA';
model.param= [0 0.001 0.01 0.1];
model.msDepth= 3;
nTrials= [5 10];
ar_band= [4 45];
ar_order_rng= 4:10;

nFiles= length(file_list);
Err= zeros(nFiles, 2*length(ar_order_rng), 2);
Class= cell(nFiles, 2);
Subject= cell(nFiles, 1);
for fi= 1:nFiles,
  file= file_list{fi,1};
  classes= file_list{fi,2};
  ii= min(find(file=='_'));
  Subject{fi}= file(1:ii-1);
  ar_ival= proc_list{fi,4};
  
  [cnt,mrk]= loadProcessedEEG(file);
  cnt= proc_selectChannels(cnt, 'not', 'E*');

  clInd= find(ismember(mrk.className, classes));
  if length(clInd)~=2,
    error('only two class problems, skipping');
  end

  cli= find(any(mrk.y(clInd,:)));
  mrk_cl= pickEvents(mrk, cli);
  Class(fi,:)= {mrk_cl.className{:}};
  fprintf('%s: <%s> vs <%s>\n', Subject{fi}, Class{fi,:});

%% RC
  fprintf('\nRC:\n');
  cnt_flt= proc_laplace(cnt, 'small', ' lap', 'filter all');
  clear cnt
  cnt_flt= proc_selectChannels(cnt_flt, ...
                               'FC3-4','C3-4','CP3-4','P3-4','PO#','O#');
  [b,a]= getButterFixedOrder(ar_band, cnt_flt.fs, 6);
  cnt_flt= proc_filt(cnt_flt, b, a);
  epo= makeSegments(cnt_flt, mrk_cl, ar_ival);
  clear cnt_flt
  
  for ai= 1:length(ar_order_rng),
    ar_order= ar_order_rng(ai);
    fv= proc_rcCoefsPlusVar(epo, ar_order);
    classy= selectModel(fv, model, [3 10 round(9/10*size(fv.y,2))]);
    fprintf('%2d> ', ar_order);
    [em,es]= doXvalidationPlus(fv, classy, nTrials, 1);
    Err(fi, ai, :)= em;
  end
  
  for ai= 1:length(ar_order_rng),
    ar_order= ar_order_rng(ai);
    fv= proc_arCoefsPlusVar(epo, ar_order);
    classy= selectModel(fv, model, [3 10 round(9/10*size(fv.y,2))]);
    fprintf('%2d> ', ar_order);
    [em,es]= doXvalidationPlus(fv, classy, nTrials, 1);
    Err(fi, length(ar_order_rng)+ai, :)= em;
  end

end

Err= reshape(Err(:,:,1), [nFiles length(ar_order_rng) 2]);

%save ar_order_sel Err ar_order_rng file_list proc_list
