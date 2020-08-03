function band= select_band(cnt, mrk, ival, varargin)
%SELECT_BAND - Select a frequency band with good descrimination
%
%Synopsis:
% BAND= select_band(CNT, MRK, IVAL, <OPT>)
%
%Description:
%The spectrum is calculated from electrodes over the sensorimotor
%areas [opt.areas] after Laplace filtering [opt.do_laplace] in the
%frequency range 5-40 Hz [opt.band] ith 1 Hz resolution. The range 
%defines the lower and upper limit of the interval that will be
%selected. Then a score is calculated for each channel and frequency
%as r-square value [opt.score_proc] of the two classes and summed
%up across all channels, resulting in a score for each frequency.
%Next the two frequencies having the two highest scores are
%selected. These define the initial band that is iteratively enlarged
%in the subsequent steps. In each step the band is enlarged either one
%Hz to the lower or to the upper side, depending on whether the sum of
%the score below the lower limit or above the upper limit is
%larger. This process is repeated until the score within the selected
%band encompasses 85% [opt.threshold] of the total score.
%
%Arguments:
% CNT  - Struct of continuous data
% MRK  - Struct of markers
% IVAL - Time interval in which the spectra should be descriminated
% OPT  - Struct or property/value list of optinal parameters, see
%    description above
%
%Returns:
% BAND - Selected frequency band
%
%Remark.
% This function is obsolete. Use select_bandnarrow or select_bandbroad
% instead.

% Author(s): Benjamin Blankertz

motor_areas= {{'FC5,3','CFC5,3','C5,3','CCP5,3','CP5,3'},
              {'FC1-2','CFC1-2','C1-2','CCP1-2','CP1-2'}, 
              {'FC4,6','CFC4,6','C4,6','CCP4,6','CP4,6'}};
done_laplace= ~isempty(strpatternmatch('* lap', cnt.clab));
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'band', [5 40], ...
                  'score_proc', 'r_square', ...
                  'areas', motor_areas, ...
                  'do_laplace', ~done_laplace, ...
                  'threshold', 0.85);

[score_fcn, score_param]= getFuncParam(opt.score_proc);

if opt.do_laplace,
  cnt= proc_laplace(cnt);
end
spec= makeEpochs(cnt, mrk, ival);
spec= proc_spectrum(spec, opt.band, 'db_scaled',1);
%%spec= proc_spectrum(spec, opt.band, 'db_scaled',0);  %% isn't this better?
score= feval(['proc_' score_fcn], spec, score_param{:});

%% choose good channels
chanscore= sqrt(sum(score.x.^2, 1));
for aa= 1:length(opt.areas),
  ci= chanind(score, opt.areas{aa});
  [mm,mi]= max(chanscore(ci));
  chansel(aa)= ci(mi);
end

%% choose initial band as limited by the two top score frequencies
freqscore= mean(score.x(:,chansel),2);
[ss,si]= sort(freqscore);
bandsel= sort(si(end-1:end));

%% iteratively enlarge band
thresh= opt.threshold*sum(freqscore);
while sum(freqscore(bandsel(1):bandsel(2)))<thresh,
  b= [bandsel(1)-1 bandsel(2)+1];
  if bandsel(1)==1,
    extend_dir= 2;
  elseif bandsel(2)==length(freqscore),
    extend_dir= 1;
  else
    d1= freqscore(b);
    d2= [max(freqscore(1:b(1))); max(freqscore(b(2):end))];
    dirscore= d1+d2;
    [mm,extend_dir]= max(dirscore);
  end
  bandsel(extend_dir)= b(extend_dir);
end

band= spec.t(bandsel);
