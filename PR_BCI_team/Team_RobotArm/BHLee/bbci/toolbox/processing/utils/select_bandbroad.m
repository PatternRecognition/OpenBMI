function band= select_bandbroad(cnt, mrk, ival, varargin)
%band= select_bandnarrow(cnt, mrk, ival, <opt>)

motor_areas= {{'FC5,3','CFC5,3','C5,3','CCP5,3','CP5,3'},
              {'FC1-2','CFC1-2','C1-2','CCP1-2','CP1-2'}, 
              {'FC4,6','CFC4,6','C4,6','CCP4,6','CP4,6'}};
done_laplace= ~isempty(strpatternmatch('* lap', cnt.clab));
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'band', [5 35], ...
                  'score_proc', 'r_square_signed', ...
                  'areas', motor_areas, ...
                  'do_laplace', ~done_laplace, ...
                  'smooth_spectrum', 1, ...
                  'threshold', 1/3, ...
                  'threshold_total', 0.8, ...
                  'threshold_valley', 0.1);

[score_fcn, score_param]= getFuncParam(opt.score_proc);

if opt.do_laplace,
  cnt= proc_laplace(cnt);
end
cnt= proc_selectChannels(cnt, cat(2, opt.areas{:}));
spec= makeEpochs(cnt, mrk, ival);
spec= proc_spectrum(spec, opt.band, 'db_scaled',1);
%%spec= proc_spectrum(spec, opt.band, 'db_scaled',0);  %% isn't this better?
score= feval(['proc_' score_fcn], spec, score_param{:});
if opt.smooth_spectrum,
  score.x= movingAverage(score.x, 3, 'method','centered', 'window',[.5 1 .5]');
end

%% for each channel calc a positive (class 1 > class 2) and a
%% negative score (class 2 > class 1)
[nFreqs, nChans]= size(score.x);
scp= reshape(max(0, score.x), [nFreqs, nChans]);
scn= reshape(max(0, -score.x), [nFreqs, nChans]);
for cc= 1:nChans,
  disc= find(scp(:,cc)<0.1*max(scp(:,cc)));
  scp(disc,cc)= 0;
  disc= find(scn(:,cc)<0.1*max(scn(:,cc)));
  scn(disc,cc)= 0;
end

%% calculate score for each channel and choose good channels
%% (one from each area)
chanscp= sqrt(sum(scp.^2, 1));
chanscn= sqrt(sum(scn.^2, 1));
chanscore= max(chanscp, chanscn);
for aa= 1:length(opt.areas),
  ci= chanind(score, opt.areas{aa});
  [mm,mi]= max(chanscore(ci));
  chansel(aa)= ci(mi);
end

%% determine favorable narrow band
freqscore= max(scp, scn);
freqscore= mean(freqscore(:,chansel),2);
[topscore,mm]= max(freqscore);
mm= max(2, min(length(freqscore)-1, mm));
bidx= [mm-1:mm+1];

%% determine for each channel the favorable sign of class difference
%% and adjust score accordingly
freqscore= score.x(:,chansel);
for aa= 1:length(chansel),
  if mean(scp(bidx,chansel(aa))) < mean(scn(bidx,chansel(aa))),
    freqscore(:,aa)= -score.x(:,chansel(aa));
  end
end
freqscore= mean(freqscore, 2);

%% choose initial band as the top score frequency
[topscore,mm]= max(freqscore);
bandsel= [mm mm];

%% iteratively enlarge band
goon= 1;
absscore= max(freqscore, 0);
thresh= opt.threshold_total*sum(absscore);
while goon & sum(absscore(bandsel(1):bandsel(2)))<thresh,
  b= [max(1,bandsel(1)-1) min(length(freqscore),bandsel(2)+1)];
  narrowscore= absscore(b);
  broadscore= [max(absscore(1:b(1))); max(absscore(b(2):end))];
  if bandsel(1)==1,
    extend_dir= 2;
    narrowscore(1)= -inf;
    broadscore(1)= -inf;
  elseif bandsel(2)==length(freqscore),
    extend_dir= 1;
    narrowscore(2)= -inf;
    broadscore(2)= -inf;
  end
  [ma, extend_dir]= max(narrowscore);
  %% if narrowscore for one direction is good enough, take it,
  %% otherwise take broadscore into account.
  if ma < opt.threshold*topscore,
    dirscore= narrowscore + broadscore;
    [ma, extend_dir]= max(dirscore);
    %% check whether extension in chosen direction is allowed
    isallowed=  freqscore(b(extend_dir)) >= -opt.threshold_valley*topscore;
    if ~isallowed,
      %% if not, try the other direction
      extend_dir= 3-extend_dir;
      %% check whether it is allowed and whether it is worthwhile
      isallowed=  freqscore(b(extend_dir)) >= -opt.threshold_valley*topscore;
      isworthwhile= narrowscore(extend_dir) > .5*opt.threshold*topscore | ...
          broadscore(extend_dir) > .5*opt.threshold*topscore;
      if ~isallowed | ~isworthwhile,
        goon= 0;
      end
    end
  end
  if goon,
    bandsel(extend_dir)= b(extend_dir);
  end
end

band= spec.t(bandsel);
