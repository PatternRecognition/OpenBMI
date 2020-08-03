function band= select_bandnarrow(cnt, mrk, ival, varargin)
%SELECT_BANDNARROW - Select a narrow frequency band with good descrimination
%
%Synopsis:
% BAND= select_bandnarrow(CNT, MRK, IVAL, <OPT>)
%
%Description:
%The spectrum is calculated from electrodes over the sensorimotor
%areas [opt.areas] after Laplace filtering [opt.do_laplace; double
%laplace filtering is avoided] in the
%frequency range 5-35 Hz [opt.band] ith 1 Hz resolution. The range 
%defines the lower and upper limit of the interval that will be
%selected. Then a score is calculated for each channel and frequency
%as signed r-square value [opt.score_proc] of the two classes. The
%the score is smoothed [opt.smooth_spectrum] by a (centered) moving
%average in a 3 Hz window.
%The frequency having the highest absolute score is selected. If the
%score at this frequency is negative for a channel, the score of that
%channel is multiplied by -1. After this correction of the sign,
%the objective is to select a frequency band that collects most of
%the high scores while avoiding negative scores.
%(Having an ERD in one channel and ERS in another one is fine, but
%an ERD in one part of the frequency band and an ERS in other part
%within the same channel is bad, because these effects whould cancel
%out, when calculating the band power.)
%In the following 'score' denotes the sign-corrected score averaged
%across all channels. The initial band is defined by the frequency 
%that has this highest score. First the lower limit of the band is 
%decreased as long as the corresponding scores are at least 1/3 
%[opt.threshold] of the topscore. Then the upper limit is increased
%in the same manner.
%[There is some more heuristic for the borders of the band, but I 
%will not describe them here.]
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
% Author(s): Benjamin Blankertz
%            06-12 Javier Pascual. Modified to allow subsets of electrodes

motor_areas= {{'FC5,3','CFC5,3','C5,3','CCP5,3','CP5,3'},
              {'FC1-2','CFC1-2','C1-2','CCP1-2','CP1-2'}, 
              {'FC4,6','CFC4,6','C4,6','CCP4,6','CP4,6'}};
done_laplace= ~isempty(strpatternmatch('* lap*', cnt.clab));
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'band', [5 35], ...
                  'band_topscore', [7 35], ...
                  'score_proc', 'r_square_signed', ...
                  'areas', motor_areas, ...
                  'do_laplace', ~done_laplace, ...
                  'laplace_require_neighborhood', 1,...
                  'smooth_spectrum', 1, ...
                  'threshold', 1/3, ...
                  'threshold_stop', 1/20, ...
                  'threshold_ext', 1/2);

if isempty(opt.areas),
  opt.areas= {cnt.clab};
end

% Sanity check: interval "band_topscore" must be within interval "band"
if opt.band_topscore(1) < opt.band(1)
  opt.band_topscore(1) = opt.band(1)+1;
  warning(sprintf('opt.band_topscore automatically adapted in select_bandnarrow: [%d %d]\n', opt.band_topscore));
end
if opt.band_topscore(2) > opt.band(2)
  opt.band_topscore(2) = opt.band(2)-1;
  warning(sprintf('opt.band_topscore automatically adapted in select_bandnarrow: [%d %d]\n', opt.band_topscore));
end

[score_fcn, score_param]= getFuncParam(opt.score_proc);

if opt.do_laplace,
  cnt= proc_laplacian(cnt,'require_complete_neighborhood', opt.laplace_require_neighborhood);
end
if diff(ival)>=1000,                      
  winlen= cnt.fs;                                   
  spec_ival= ival;                              
else                                                
  winlen= cnt.fs/2;                                 
  if diff(ival)<500,                            
%     bbci_bet_message('Enlarging interval to calculate spectra\n');
    spec_ival=  mean(ival) + [-250 250];                      
  else                                                            
    spec_ival= ival;                                          
  end                                                             
end                                                               
spec= cntToEpo(cnt, mrk, spec_ival, 'clab', cat(2, opt.areas{:}));
spec= proc_spectrum(spec, opt.band, 'win',kaiser(winlen,2));
%% or this?
%spec= proc_spectrum(spec, opt.band, 'win',kaiser(winlen,2), 'db_scaled',0);
score= feval(['proc_' score_fcn], spec, score_param{:});
if opt.smooth_spectrum,
  score.x= movingAverage(score.x, 3, 'method','centered','window',[.5 1 .5]');
end

%% choose good channels (CAUTION: pos and neg scores are summed up!)
idx= find(score.t>=opt.band_topscore(1) & ...
          score.t<=opt.band_topscore(2));
chanscore= sqrt(sum(score.x(idx,:).^2, 1));
aaa = 1;
for aa= 1:length(opt.areas),
  ci= chanind(score, opt.areas{aa});
  if(~isempty(ci)),
     [mm,mi]= max(chanscore(ci));
     chansel(aaa)= ci(mi);
     aaa = aaa + 1;
  end;
end

%% choose initial band as limited by the top score frequency
tempscore= mean(abs(score.x(idx,chansel)),2);
[topscore,topidx]= max(tempscore);
topfreq= idx(topidx);

%% correct sign and redo
xx= score.x(:,chansel);
for ci= 1:length(chansel),
  sgn= sign(xx(topfreq,ci));
  xx(:,ci)= xx(:,ci) * sgn;
end
freqscore= mean(xx,2);
[topscore,topfreq]= max(freqscore);
bandsel= [topfreq topfreq];
bandext= [0 0];

%% iteratively enlarge band
goon= 1;
while goon & bandsel(1)>1,
  fsc= freqscore(bandsel(1)-1);
  if fsc >= topscore*opt.threshold,
    bandsel(1)= bandsel(1)-1;
  else
    goon= 0;
    if fsc < topscore*opt.threshold_stop,
      bandext(1)= 0;
    elseif mean(freqscore(bandsel(1)+[-1 0])) >= topscore*opt.threshold_ext,
      bandext(1)= -1;
    else
      bandext(1)= -0.5;
    end
  end
end
goon= 1;
while goon & bandsel(2)<length(freqscore),
  fsc= freqscore(bandsel(2)+1);
  if fsc >= topscore*opt.threshold,
    bandsel(2)= bandsel(2)+1;
  else
    goon= 0;
    if fsc < topscore*opt.threshold_stop,
      bandext(2)= 0;
    elseif mean(freqscore(bandsel(2)+[0 1])) >= topscore*opt.threshold_ext,
      bandext(2)= 1;
    else
      bandext(2)= 0.5;
    end
  end
end

band= spec.t(bandsel) + bandext;

