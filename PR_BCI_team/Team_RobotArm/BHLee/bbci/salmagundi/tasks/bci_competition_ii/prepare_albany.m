data_dir= [DATA_DIR 'eegImport/bci_competition_ii/albany/'];

clab= {'FC5','FC3','FC1','FCz','FC2','FC4','FC6', ...
       'C5','C3','C1','Cz','C2','C4','C6', ...
       'CP5','CP3','CP1','CPz','CP2','CP4','CP6', ...
       'Fp1','Fpz','Fp2', 'AF7','AF3','AFz','AF4','AF8', ...
       'F7','F5','F3','F1','Fz','F2','F4','F6','F8', ...
       'FT7','FT8','T7','T8','T9','T10','TP7','TP8', ...
       'P7','P5','P3','P1','Pz','P2','P4','P6','P8', ...
       'PO7','PO3','POz','PO4','PO8','O1','Oz','O2','Iz'};

clear cnt mrk
subject_list= {'AA','BB','CC'};
for ss= 1:length(subject_list),
  subject= subject_list{ss};
  cnt.x= zeros(0, 64);
  cnt.clab= clab;
  cnt.fs= 160;
  cnt.title= sprintf('albany %s', subject);
  tc= zeros(0, 1);
  rc= zeros(0, 1);
  if ss<3,
    run_list= 1:6;
  else
    run_list= [1 3:6];  %% second run was corrupt
  end
  for rr= run_list,
    file= sprintf('%s%03d', subject_list{ss}, rr),
    load([data_dir file]);
    cnt.x= cat(1, cnt.x, signal);
    
    tc= cat(1, tc, TargetCode);
    rc= cat(1, rc, ResultCode);
  end
  pos= find(diff([0; tc])>0);
  mrk.pos= pos';
  mrk.toe= tc(pos)';
  mrk.pos_eot= find(diff([0; rc])>0);
  mrk.res= rc(mrk.pos_eot)';
  mrk.y= [mrk.toe==1; mrk.toe==2; mrk.toe==3; mrk.toe==4];
  mrk.fs= cnt.fs;
  mrk.className= {'top','upper','lower','bottom'};

  mnt= setElectrodeMontage(clab);
  grd= sprintf('F3,Fz,F4\nC3,Cz,C4\nP3,legend,P4');
  mnt= setDisplayMontage(mnt, grd);

  save_file= sprintf('albany_%s_train', subject_list{ss});
  saveProcessedEEG(['bci_competition_ii/' save_file], cnt, mrk, mnt);
end



clear cnt mrk
run_list= 7:10;
for ss= 1:length(subject_list),
  subject= subject_list{ss};
  cnt.x= zeros(0, 64);
  cnt.clab= clab;
  cnt.fs= 160;
  cnt.title= sprintf('albany %s test data', subject);
  tr= zeros(0, 1);
  for rr= run_list,
    file= sprintf('%s%03d', subject, rr),
    load([data_dir file]);
    cnt.x= cat(1, cnt.x, signal);
    
    if sum(trial==max(trial))<sum(trial==1),  %% delete incomplete last trials
      trial(find(trial==max(trial)))= -1;
    end
    tr= cat(1, tr, trial);
  end
  pos= find(diff([-1; tr])>0) + 176;  %% time of feedback presentation
  mrk.pos= pos';
  mrk.fs= cnt.fs;
  mrk.className= {'top','upper','lower','bottom'};

  mnt= setElectrodeMontage(cnt.clab);
  grd= sprintf('F3,Fz,F4\nC3,Cz,C4\nlegend,Pz,P8');
  mnt= setDisplayMontage(mnt, grd);

  save_file= sprintf('albany_%s_test', subject);
  saveProcessedEEG(['bci_competition_ii/' save_file], cnt, mrk, mnt);
end



clear cnt mrk
run_list= 7:10;
for ss= 1:length(subject_list),
  subject= subject_list{ss};
  cnt= struct('clab',{clab});
  cnt.x= zeros(0, 64);
  cnt.fs= 160;
  cnt.title= sprintf('albany %s test data', subject);
  pos= zeros(0, 1);
  lab= zeros(1, 0);
  pred= zeros(1, 0);
  their_trialnr= zeros(1, 0);
  our_trialnr= zeros(1, 0);
  for rr= run_list,
    file= sprintf('%s%03d', subject, rr),
    load([data_dir file]);
    
    if sum(trial==max(trial))<sum(trial==1),  %% delete incomplete last trials
      trial(find(trial==max(trial)))= -1;
    end
%    ifb= find(Feedback==1);
%    pp= ifb(find(diff([-1; trial(ifb)])>0));  %% time of feedback presentation
    nTrials= length(pos);
    pp= find(diff([-1; trial])>0) + 176;  %% time of feedback presentation
    pos= cat(1, pos, size(cnt.x,1)+pp);
    cnt.x= cat(1, cnt.x, signal);
    tn= trial(pp);
    our_trialnr= cat(2, our_trialnr, nTrials+tn');

    labels= load([data_dir file 'LABELS.dat']);
    lab= cat(2, lab, labels(:,3)');
    pred= cat(2, pred,  labels(:,4)');
    trialnr= labels(:,2);
    their_trialnr= cat(2, their_trialnr, nTrials+trialnr');
  end
  mrk= struct('fs', cnt.fs);
  mrk.className= {'top','upper','lower','bottom'};
  ttt= unique(our_trialnr);
  mrk.pos= zeros(1, length(ttt));
  mrk.y= zeros(4, length(ttt));
  mrk.albany_pred= zeros(1, length(ttt));
  for ii= 1:length(ttt),
    oti= min(find(our_trialnr==ttt(ii)));
    tti= min(find(their_trialnr==ttt(ii)));
    if ~isempty(tti),
      mrk.pos(ii)= pos(oti);
      mrk.y(lab(tti),ii)= 1;
      mrk.albany_pred(ii)= pred(tti);
    end
  end
  valid= find(mrk.pos>0);
  mrk.indexedByEpochs= {'albany_pred'};
  mrk= mrk_selectEvents(mrk, valid);
  
  mnt= setElectrodeMontage(cnt.clab);
  grd= sprintf('F3,Fz,F4\nC3,Cz,C4\nlegend,Pz,P8');
  mnt= setDisplayMontage(mnt, grd);

  save_file= sprintf('albany_%s_test_withLabels', subject);
  saveProcessedEEG(['bci_competition_ii/' save_file], cnt, mrk, mnt);
end
