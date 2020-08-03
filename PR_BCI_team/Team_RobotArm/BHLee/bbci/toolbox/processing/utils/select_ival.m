function ival= select_ival(cnt, mrk, varargin)
%SELECT_IVAL - Select a time interval with good descrimination
%
%Synopsis:
% IVAL= select_ival(CNT, MRK, <OPT>)
%
%Description:
%This function is called with band-pass filtered signals, which are
%Laplace filtered [opt.do_laplace] and then the envelope is calculated.
%Then a score is calculated for each channel and time point
%as r-square value [opt.score_proc] of the two classes and summed
%up across all channels, resulting in a score for each time point.
%Next the two time points having the two highest scores are
%selected. These define the initial interval that is iteratively enlarged
%in the subsequent steps. In each step the interval is enlarged either one
%sample to the lower or to the upper side, depending on whether the sum of
%the score below the lower limit or above the upper limit is
%larger. This process is repeated until the score within the selected
%time interval encompasses 85% [opt.threshold] of the total score.
%
%Arguments:
% CNT - Struct of continuous data
% MRK - Struct of markers
% OPT - Struct or property/value list of optinal parameters, see
%    description above
%
%Returns:
% IVAL - Selected time interval
%
%Remark.
% This function is obsolete. Use select_timeival instead.

% Author(s): Benjamin Blankertz

motor_areas= {{'FC5,3','CFC5,3','C5,3','CCP5,3','CP5,3'},
              {'FC1-2','CFC1-2','C1-2','CCP1-2','CP1-2'}, 
              {'FC4,6','CFC4,6','C4,6','CCP4,6','CP4,6'}};
done_laplace= ~isempty(strpatternmatch('* lap', cnt.clab));
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'filter', [], ...
                  'max_ival', [250 5000], ...
                  'score_proc', 'r_square', ...
                  'areas', motor_areas, ...
                  'do_laplace', ~done_laplace, ...
                  'threshold', 0.85);

[score_fcn, score_param]= getFuncParam(opt.score_proc);

erd= cnt;
if opt.do_laplace,
  erd= proc_laplace(erd);
end
if ~isempty(opt.filter),
  erd= proc_filt(erd, opt.filter.b, opt.filter.a);
end
erd= proc_rectifyChannels(erd);
erd= proc_movingAverage(erd, 200);
erd= makeEpochs(erd, mrk, [-500 opt.max_ival(2)]);
erd= proc_baseline(erd, [-500 0]);  %% this is debateable
erd= proc_selectIval(erd, opt.max_ival);

score= feval(['proc_' score_fcn], erd, score_param{:});

%% choose good channels
chanscore= sqrt(sum(score.x.^2, 1));
for aa= 1:length(opt.areas),
  ci= chanind(score, opt.areas{aa});
  [mm,mi]= max(chanscore(ci));
  chansel(aa)= ci(mi);
end

%% choose initial ival as limited by the two top score time points
timescore= mean(score.x(:,chansel),2);
[ss,si]= sort(timescore);
ivalsel= sort(si(end-1:end));

%% iteratively enlarge ival
thresh= opt.threshold*sum(timescore);
while sum(timescore(ivalsel(1):ivalsel(2)))<thresh,
  b= [ivalsel(1)-1 ivalsel(2)+1];
  if ivalsel(1)==1,
    extend_dir= 2;
  elseif ivalsel(2)==length(timescore),
    extend_dir= 1;
  else
    d1= timescore(b);
    d2= [max(timescore(1:b(1))); max(timescore(b(2):end))];
    dirscore= d1+d2;
    [mm,extend_dir]= max(dirscore);
  end
  ivalsel(extend_dir)= b(extend_dir);
end

ival= erd.t(ivalsel);
