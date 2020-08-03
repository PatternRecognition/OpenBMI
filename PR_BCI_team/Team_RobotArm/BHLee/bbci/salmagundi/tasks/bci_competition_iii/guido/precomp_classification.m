su = 1;  % SUBJECT NUMBER

% GET THE DATA
file_dir= [EEG_IMPORT_DIR 'bci_competition_iii/martigny/'];

epo = struct('className',{{'left','right','word'}},'x',[],'y',[],'block',[]);
for p = 1:3;
S = load([file_dir 'train_subject' int2str(su) '_psd0' int2str(p) '.mat']);
epo.x = cat(2,epo.x,S.X');
epo.y = cat(2,epo.y,S.Y');
epo.block = cat(2,epo.block,p*ones(size(S.Y')));
end

epo.title = untex(S.nfo.name(1:end-2));

epo.y = double([epo.y==2;epo.y==3;epo.y==7]);



% SET DEFAULTS
opt_xv= struct('out_trainloss',1, 'outer_ms',1, 'xTrials',[10 10],'msTrials',[3 10 -1],...
               'verbosity',3);
opt2 = opt_xv;
opt2 = rmfield(opt2,'xTrials');
opt2.divTr = {{find(epo.block<max(epo.block))}};
opt2.divTe = {{find(epo.block==max(epo.block))}};
opt2.msTrials = [3 10];


model_RLDA= struct('classy', 'RLDA');
model_RLDA.param= [0 0.01 0.1 0.3 0.5 0.7];
model_LDA= 'LDA';


% XVAL
xvalidation(epo, model_LDA, opt_xv);

% CAUSAL
xvalidation(epo, model_RLDA, opt2);




