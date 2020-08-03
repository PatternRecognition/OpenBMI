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
model.param= [0 0.001 0.005 0.01 0.05 0.1 0.5];
model.msDepth= 3;
nTrials= [5 10];
ar_band= [4 45];

nFiles= length(file_list);
Err= zeros(nFiles, 3, 2);
Class= cell(nFiles, 2);
for fi= 1:nFiles,
  file= file_list{fi,1};
  classes= file_list{fi,2};
  [csp_band, csp_ival, csp_nPat]= deal(proc_list{fi,1:3});
  [ar_ival, ar_order]= deal(proc_list{fi,4:5});
  
  [cnt,mrk]= loadProcessedEEG(file);
  cnt= proc_selectChannels(cnt, 'not', 'E*');

  clInd= find(ismember(mrk.className, classes));
  if length(clInd)~=2,
    warning('only two class problems, skipping');
    continue;
  end

  cli= find(any(mrk.y(clInd,:)));
  mrk_cl= pickEvents(mrk, cli);
  Class(fi,:)= {mrk_cl.className{:}};
  fprintf('%s: <%s> vs <%s>\n', file, Class{fi,:});

%% RC
  fprintf('\nRC:\n');
  cnt_flt= proc_laplace(cnt, 'small', ' lap', 'filter all');
  clear cnt
  cnt_flt= proc_selectChannels(cnt_flt, ...
                               'FC3-4','C3-4','CP3-4','P3-4','PO#','O#');
  [b,a]= getButterFixedOrder(ar_band, cnt_flt.fs, 6);
  cnt_flt= proc_filt(cnt_flt, b, a);
  fv= makeSegments(cnt_flt, mrk_cl, ar_ival);
  clear cnt_flt
  fv= proc_rcCoefsPlusVar(fv, ar_order);
  classy= selectModel(fv, model, [3 10 round(9/10*size(fv.y,2))]);
  [em,es]= doXvalidationPlus(fv, classy, nTrials, 1);
  Err(fi, 1, :)= em;

%% CSP
  fprintf('\nCSP:\n');
  cnt= loadProcessedEEG(file);
  [b,a]= getButterFixedOrder(csp_band, cnt.fs, 6);
  cnt_flt= proc_filt(cnt, b, a);
  clear cnt
  
  fv= makeSegments(cnt_flt, mrk_cl, csp_ival);
  clear cnt_flt
  fv= proc_selectChannels(fv, 'not', 'Fpz','AF#','T#');
  fv.proc=['fv= proc_csp(epo, ' int2str(csp_nPat) '); ' ...
           'fv= proc_variance(fv); '];
  
  classy= selectModel(fv, model, [3 10 round(9/10*size(fv.y,2))]);
  [em,es]= doXvalidationPlus(fv, classy, nTrials, 1);
  Err(fi, 2, :)= em;

  
%% RC on CSP
  fprintf('\nRC on CSP:\n');
  [W,W]= proc_csp(fv, csp_nPat);
  csp_input_chans= chanind(cnt, 'not', 'Fpz','AF#','T#');
  
  cnt= loadProcessedEEG(file);
  clear cnt
  cnt_pr= proc_selectChannels(cnt, csp_input_chans);
  [b,a]= getButterFixedOrder(ar_band, cnt.fs, 6);
  cnt_pr= proc_filt(cnt_pr, b, a);
  cnt_pr= proc_linearDerivation(cnt_pr, W);
  fv= makeSegments(cnt_pr, mrk_cl, ar_ival);
  clear cnt_pr
  fv= proc_rcCoefsPlusVar(fv, ar_order);

  classy= selectModel(fv, model, [3 10 round(9/10*size(fv.y,2))]);
  [em,es]= doXvalidationPlus(fv, classy, nTrials, 1);
  Err(fi, 3, :)= em;

  fprintf('\n%s: <%s> vs <%s>:  %2.1f%%  %2.1f%%  %2.1f%%\n\n', ...
          file, Class{fi,:}, Err(fi, :, 1));

end


for fi= 1:nFiles,
    fprintf('%40s: <%8s> vs <%8s>:  %2.1f%%  %2.1f%%  %2.1f%%\n', ...
          file_list{fi,1}, Class{fi,:}, Err(fi, :, 1));
end
