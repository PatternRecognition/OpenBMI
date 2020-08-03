file_no= 1;

file_list= {'Matthias_04_03_24/imag_cursMatthias', {'left','foot'}; ...
            'Matthias_04_03_24/imag_cursMatthias', {'left','right'}; ...
            'Matthias_04_03_03/imagMatthias', {'left','foot'}; ...
            'Guido_04_03_29/imag_cursGuido', {'left','right'}; ...
            strcat('Guido_04_03_29/imag_', {'move','lett'}, 'Guido'), {'left','right'}; ...
            'Falk_04_03_31/imag_cursFalk', {'left','right'}; ...
            'Falk_04_03_31/imag_cursFalk', {'left','foot'}; ...
            'Falk_04_03_31/imag_cursFalk', {'right','foot'}; ...
            'Klaus_04_04_08/imag_curs', {'right','foot'}};

%for file_no= 1:length(file_list),

file= file_list{file_no,1};
classes= file_list{file_no,2};

csp.clab= {'not','E*','Fp*','AF*','I*','OI*',...
           'OPO*','TP9,10','T9,10','FT9,10'};b
csp.ival= [750 3500];
csp.band= [7 32];
csp.filtOrder= 5;
csp.nPat= 3;

opt_xv= struct('sample_fcn','leaveOneOut', ...
               'save_proc_params',{{'csp_w','csp_la','csp_a'}});

[cnt, mrk, mnt]= eegfile_loadMatlab(file);
mrk= mrk_selectClasses(mrk, classes);

[b,a]= butter(csp.filtOrder, csp.band/cnt.fs*2);
cnt_flt= proc_filt(cnt, b, a);
epo= makeEpochs(cnt_flt, mrk, csp.ival);
epo= proc_selectChannels(epo, csp.clab);

%% First we demonstrate the WRONG VALIDATION!!!
%% The application of csp before the cross-validation use label information
%% of trials which become test trials in the subsequent cross-validation.
%% In order to be unbiased, the test trials in the cross-validation have
%% to be completely 'unseen' (i.e., unused by the algorithm). This
%% principle is violated here and causes and underestimation of the
%% generalization error.
[fv, csp_w, csp_la]= proc_csp(epo, csp.nPat);
fv= proc_variance(fv);
fv= proc_logarithm(fv);
[loss(file_no,1), loss_std, out0]= ...
    xvalidation(fv, 'LDA', opt_xv);

mt= mnt_adaptMontage(mnt, epo);
figure(1);
plotCSPatterns(fv, mt, csp_w, csp_la);

%% Here comes the correct way: CSP is caluclated *within* the 
%% cross-validation on each training set (proc.train). The calculated
%% filters (projection matrix csp_w) are saved (proc.memo) and
%% can be used for projecting the test samples (proc.apply).
proc= struct('memo', {{'csp_w','csp_la','csp_a'}});
proc.train= ['[fv,csp_w,csp_la,csp_a]= proc_csp2(fv, ' ...
             int2str(csp.nPat) '); ' ...
             'fv= proc_variance(fv); ' ...
             'fv= proc_logarithm(fv);'];
proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ' ...
             'fv= proc_variance(fv); ' ...
             'fv= proc_logarithm(fv);'];
[loss(file_no,2), loss_std, out, memo]= ...
    xvalidation(epo, 'LDA', opt_xv, 'proc',proc);

iserr0= sign(out0)~=[-1 1]*fv.y;
iserr= sign(out)~=[-1 1]*fv.y;

%% errors in leave-one-out validation, that were correct in
%% global csp classification:
isnewerr= iserr & ~iserr0;

iBad= find(isnewerr);
iGood= find(~iserr & ~iserr0);
show_trials= iBad(1:min(5,length(iBad)));
show_trials= [show_trials iGood(1:10-length(show_trials))];
%show_trials= 1:10;
show_pats= 1:2*csp.nPat;

opt_scalp= struct('scalePos','none', 'contour',0, 'crossSize',1);
col= [0 0.7 0; 1 0 0];
figure(2); clf;
for tt= 1:length(show_trials),
  it= show_trials(tt);
  for pp= 1:length(show_pats),
    ii= ii+1;
    ip= show_pats(pp);
    subplotxl(length(show_pats), length(show_trials), [pp tt], 0, [0.04 0 0]);
    scalpPlot(mt, memo(it).csp_w(:,ip), opt_scalp, ...
              'linespec', {'color',col(isnewerr(it)+1,:)});
    if tt==1,
      ylabel(fv.clab{ip}, 'visible','on');
    end
  end
end
