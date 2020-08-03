subject= 'Gabriel';
dateStr= '03_05_21';

patx= [0.25 1];
paty= [0 0.75];
bk_ival= [-100 800];
bk_entry= 250;
fb_opt= struct('xlim',[-2 2], 'ylim',[-1.5 1.5], ...
               'patch_x',patx, 'patch_y',paty);

sub_dir= [subject '_' dateStr '/'];
fig_dir= ['preliminary/' sub_dir];

train_file= [sub_dir 'selfpaced2s' subject];

method= struct('ilen', 1270);
method.clab= {'CFC3,4', 'C5-6', 'CCP5-6', 'CP5,3,4,6','P3,4'};
method.proc= ['fv= proc_filtBruteFFT(fv, [0.8 3], 128, 200); ' ...
              'fv= proc_jumpingMeans(fv, 5);'];
method.jit= [-50, -150];
method.model= 'MSR';

method_fade= method;
method_fade.model= {'RDAreject', 'gamma',0, ...
                    'rejectMethod','classCond', 'rejectParam',[0.9 0.1]};

method_dtct= struct('ilen', 1270);
method_dtct.clab= {'CFC3,4', 'C5-6', 'CCP5-6', 'CP5,3,4,6','P3,4'};
method_dtct.proc= ['fv= proc_filtBruteFFT(epo, [0.8 3], 128, 200); ' ...
                   'fv= proc_jumpingMeans(fv, 5);'];
method_dtct.jit= [-50, -150];
method_dtct.jit_noevent= [-900, -1000];
method_dtct.model= 'linearPerceptron';
method_dtct.combinerFcn= inline('max(x([1,2],:))-x(3,:)');


%% load data, calculate features
%% and train determination classifiers
[cnt,mrk,mnt]= loadProcessedEEG(train_file);

epo= makeEpochs(cnt, mrk, [-method.ilen 0], method.jit);
fv= proc_selectChannels(epo, method.clab);
fv= proc_applyProc(fv, method.proc);

C= trainClassifier(fv, method.model);
method.C= C;

C= trainClassifier(fv, method_fade.model);
method_fade.C= C;
%% determine upper bound for reject outputs
out= applyClassifier(fv, method_fade.model, method_fade.C);
hits= find(sign(out)==[-1 1]*fv.y);
[so,si]= sort(abs(out(hits)));
upper= so(round(0.95*length(so)));

%% extract features and train detect classifier
epo= makeEpochs(cnt, mrk, [-method_dtct.ilen 0], method_dtct.jit);
no_moto= makeEpochs(cnt, mrk, [-method_dtct.ilen 0], method_dtct.jit_noevent);
no_moto.y= ones(1,size(no_moto.y,2));
no_moto.className= {'no event'};
epo= proc_appendEpochs(epo, no_moto);
clear no_moto
fv= proc_selectChannels(epo, method_dtct.clab);
fv= proc_applyProc(fv, method_dtct.proc);
C= trainClassifier(fv, method_dtct.model);
method_dtct.C= C;


%% output on all samples of the training file
%fb= calc_classifier_output(cnt, method, method_dtct);
%ff= calc_classifier_output(cnt, method_fade);

%% output on all samples of a test file
fi= 3;  %% ... or 6 7 8 10
test_file= [sub_dir 'selfpaced2s_fb' int2str(fi) subject];
[cnt, mrk]= loadProcessedEEG(test_file);
fb= calc_classifier_output(cnt, method, method_dtct);
ff= calc_classifier_output(cnt, method_fade);


if isstruct(mrk) & ~isempty(mrk.pos),
%% remove intervals around keypress
  iv= getIvalIndices(bk_ival, fb.fs);
  T= length(iv);
  nEvents= length(mrk.pos);
  IV= iv(:)*ones(1,nEvents) + ones(T,1)*mrk.pos;
  fb.x(IV, :)= NaN;
  
  ival= mrk.pos([1 end])/mrk.fs*1000;
  fb= proc_selectIval(fb, ival + [-1000 1000]);
  ff= proc_selectIval(ff, ival + [-1000 1000]);
end

fb= proc_subsampleByLag(fb, 4);
ff= proc_subsampleByLag(ff, 4);

veto_fb_plot(fb, fb_opt, 'strength',abs(ff.x), 'upper_bound', fade_upper);



return





title(fb.title);

hitL= -fb.x(:,1)>=patx(1) & fb.x(:,2)>=paty(1) & ...
     (fb.x(:,2)>=paty(2)+(patx(1)+fb.x(:,1))*diff(paty)/diff(patx));
hitR= fb.x(:,1)>=patx(1) & fb.x(:,2)>=paty(1) & ...
     (fb.x(:,2)>=paty(2)+(patx(1)-fb.x(:,1))*diff(paty)/diff(patx));
str= sprintf('%.1f%%  [%.1f]', 100*mean(hitL), median(st(find(hitL))));
text(fb_opt.xlim(1)+0.02*diff(fb_opt.xlim), fb_opt.ylim(2), str, ...
     'verticalAli','top', 'fontSize',14);
str= sprintf('[%.1f]  %.1f%%', median(st(find(hitR))), 100*mean(hitR));
text(fb_opt.xlim(2)-0.02*diff(fb_opt.xlim), fb_opt.ylim(2), str, ...
     'verticalAli','top', 'horizontalAli','right', 'fontSize',14);
%%saveFigure([fig_dir 'veto_fb_plot_' int2str(kk)], [20 15]);
