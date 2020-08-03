file_list= {'Thorsten_02_07_31/imagmultimodalThorsten', {'leg','auditory'}; ...
            'Steven_02_08_12/imagmultiSteven', {'left','right'}; ...
            'Steven_02_08_12/imagmultiSteven', {'left','auditory'}; ...
            'Steven_02_08_12/imagmultiSteven', {'left','tactile'}; ...
            'Soeren_02_08_13/imagmultiSoeren', {'left','right'}; ...
            'Soeren_02_08_13/imagmultiSoeren', {'left','foot'}; ...
            'Soeren_02_08_13/imagmultiSoeren', {'left','tactile'}};

proc_list= {[7 14], [ 500 3000], 4,  [ 500 3000], 6; ...
            [7 14], [ 500 2500], 4,  [ 500 3000], 6; ...
            [7 14], [ 500 3500], 4,  [ 500 3500], 6; ...
            [7 30], [ 500 3500], 4,  [ 500 2500], 6; ...
            [7 14], [ 500 2000], 2,  [ 500 2000], 6; ...
            [7 14], [ 500 2500], 4,  [ 500 2500], 6; ...
            [7 14], [ 500 2000], 2,  [ 500 2000], 6};

model= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model.param= [0 0.001 0.01 0.1 0.5];
nTrials= [5 10];
ar_band= [4 45];

nFiles= length(file_list);
Err= zeros(nFiles, 3, 2);
Class= cell(nFiles, 2);
for fi= 1:nFiles,
  file= file_list{fi,1};
  classes= file_list{fi,2};
  ii= min(find(file=='_'));
  Subject{fi}= file(1:ii-1);
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
  fprintf('\n%s: <%s> vs <%s>\n', Subject{fi}, Class{fi,:});
  epo= makeSegments(cnt, mrk_cl, [-1500 4000]);
  clear cnt
  
  if isempty(strmatch('auditory', Class(fi,:))),
    ar_chans= {'FC3-4','C3-4','CP3-4','P3-4','PO#','O#'};
    csp_chans= {'not', 'Fpz','AF#','T#'};
  else
    ar_chans= {'FC3-4','C3-4','CP3-4','T#','P3-4','PO#','O#'};
    csp_chans= {'not', 'Fpz','AF#'};
  end
  
%% RC
  fprintf('\nRC:\n');
  fv= proc_laplace(epo, 'small', ' lap', 'filter all');
  fv= proc_selectChannels(fv, ar_chans);
  [b,a]= getButterFixedOrder(ar_band, fv.fs, 6);
  fv= proc_filt(fv, b, a);
  fv= proc_selectIval(fv, ar_ival);
  fv= proc_rcCoefsPlusVar(fv, ar_order);
  classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
  [em,es]= doXvalidationPlus(fv, classy, nTrials, 1);
  Err(fi, 1, :)= em;

%% CSP (globally calculated: cheat)
  fprintf('\nCSP global:\n');
  fv= proc_selectChannels(epo, csp_chans);
  [b,a]= getButterFixedOrder(csp_band, fv.fs, 6);
  fv= proc_filt(fv, b, a);
  fv= proc_selectIval(fv, csp_ival);
  [fv,W]= proc_csp(fv, csp_nPat);
  fv= proc_variance(fv);
  
%  classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
  classy= 'LDA';
  [em,es]= doXvalidationPlus(fv, classy, nTrials, 1);
  Err(fi, 2, :)= em;
  
%% RC on global CSPs
  fprintf('\nRC on CSP global:\n');
  fv= proc_selectChannels(epo, csp_chans);
  fv= proc_linearDerivation(fv, W);
  [b,a]= getButterFixedOrder(ar_band, epo.fs, 6);
  fv= proc_filt(fv, b, a);
  fv= proc_selectIval(fv, ar_ival);
  
  rc_err= zeros(ar_order+1, 2);
  for nCoefs= 0:ar_order,
    fv_rc= proc_rcCoefsPlusVar(fv, ar_order, nCoefs);

%    classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
    classy= 'LDA'; 
    fprintf('%d: ', nCoefs); ...
    [em,es]= doXvalidationPlus(fv_rc, classy, nTrials, 1);
    rc_err(nCoefs+1,:)= em;
  end
  Err(fi, 3, :)= min(rc_err(3:end,1));

  fprintf('\n%s: <%s> vs <%s>:  %2.1f%%  %2.1f%%  %2.1f%%\n\n', ...
          Subject{fi}, Class{fi,:}, Err(fi, :, 1));

end


for fi= 1:nFiles,
  fprintf('%12s: <%8s> vs <%8s>:  %2.1f%%  %2.1f%%  %2.1f%%\n', ...
        Subject{fi}, Class{fi,:}, Err(fi, :, 1));
end
