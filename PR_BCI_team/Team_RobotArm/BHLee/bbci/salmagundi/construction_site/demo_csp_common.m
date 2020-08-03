file = 'Gabriel_01_10_15/imagGabriel';
%file = 'Gabriel_00_09_05/selfpaced2sGabriel';


[cnt,mrk,mnt] = loadProcessedEEG(file);

cnt = proc_selectChannels(cnt,'not','E*','O*');

cnt_flt = proc_filtForth(cnt,[8 13]);

epo = makeEpochs(cnt_flt,mrk,[500 2500]);
%epo = makeEpochs(cnt,mrk,[-1300 0]-100);

% epo = proc_filtBruteFFT(epo,[0.4 3.5],128,150);

% easiest call: (2 patterns per class)
[fv,w,la,ind] = proc_csp_common(epo,2);  %or ...(epo,2,1);
% alternatively for non-centered
%[fv,w,la,ind]  = proc_csp_common(epo,2,0);

% for the fourth argument:
% [fv,w,la,ind] = proc_csp_common(epo,2,1,'all'); % use each time
% point (ERD approach) 
% or 
% [fv,w,la,ind] = proc_csp_common(epo,2,0,'mean'); % calculate over
% time the mean (SCP approach)
% fourth argument is set by default to all if 3rd argument is 1, to
% mean if 0

% Now the interesting optimisition (silent and smooth). It's
% managed by the 5th argument. Alternatively all the fields the 5th
% argument can be given as fields in epo (these are added to these
% 5th argument)

% the field dat:
dat = makeEpochs(cnt_flt,mrk,[-500 500]);
optimise.dat = dat.x;
% or for more than two arguments:
optimise.dat{1} = dat.x;
optimise.dat{2} = [500 2500];
% choose the interval out of epo.x

%or for SCP :
%dat = makeEpochs(cnt,mrk,[-1500 -800]);
%optimise.dat{1} = dat.x;
%dat = makeEpochs(cnt,mrk,[-1500 -100]);
%optimise.dat{2} = dat.x;



% the field method
optimise.method = 'id';  % see proc_spatialprojection id is default
%or
optimise.method{1} = 'id';  % see proc_spatialprojection
optimise.method{2} = 'smooth';  % see proc_spatialprojection

% and finally the field influence
optimise.influence = 0.1;  % default
% or
optimise.influence{1} = [0.1,10]; % for highest and lowest eigenvalue 
optimise.influence{2} = [0.1,10];  


% the following things are done. The non-centered covariances of
% the optimise.dat or similar a matrix for the smooth case is
% calculated. Together with the label dependent classes these
% matrices where simultaneous diagonlised. The diagonal values of
% the additionally matrices were added (for smallest eigenvalue)
% resp. subtracted (for largest eigenvalue) (see 6th argument)
% by the given influence factors. Maybe a different method can be
% found, thus a argument in optimise.influence is possible but
% forced to a fixed value so far

% alternatively it can be given as field in epo like this
% epo.optimise = optimise;

[fv,w,la,ind] = proc_csp_common(epo,2,1,'all',optimise);
% or for SCP
%[fv,w,la,ind] = proc_csp_common(epo,2,0,'mean',optimise);


% the 6th argument:
% a string which describes from  which end eigenvalue are taken... 
% for example 'lh' takes first the lowest, then the highest, then
% the lowest ....
[fv,w,la,ind] = proc_csp_common(epo,2,1,'all',optimise,'h');
% or for SCP
%[fv,w,la,ind] = proc_csp_common(epo,2,0,'mean',optimise,'h');


% Further arguments for kernelisation. But it's not implemented for
% the constraints, and not tested so far....


