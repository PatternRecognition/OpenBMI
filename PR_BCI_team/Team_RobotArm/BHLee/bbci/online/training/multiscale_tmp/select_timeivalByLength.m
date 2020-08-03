function ivals= select_timeival(erd, mrk, varargin)
%SELECT_TIMEIVAL - Select a time interval with good descrimination
%
%Synopsis:
% IVAL= select_timeival(CNT, MRK, <OPT>)
%
%Description:
%This function is called with band-pass filtered (!!) signals. They are
%Laplace filtered [opt.do_laplace] and then the envelope is calculated
%using the Hilbert transform.
%Then a score is calculated as signed r-square value [opt.score_proc]
%of the two classes for each channel and time point of the maximal time
%interval [opt.max_ival]. These scores are splitted into positive and
%negative part, rectified (i.e. the negative part is multiplied by -1)
%and then the maximum of positive and -negative part is calculated and
%and smoothed in time by a (centered) moving average of 250ms. The
%average of this score across selected channels is calcuated. Since an
%ERD in one part and an ERS in another part of the time interval within
%one channel is undesirable, but an ERD in one channel and an ERS in
%other channel at the same time is fine, the sign of the score is
%corrected such that it is positive in each channel at the time point
%of dest descrimination.
%Next the two time points having the two highest scores are
%selected. These define the initial interval that is iteratively
%enlarged in the subsequent steps. In each step the interval is
%enlarged either one sample to the lower or to the upper side,
%depending on whether the sum of the score below the lower limit or
%above the upper limit is larger. This process is repeated until the
%score within the selected time interval encompasses 80%
%[opt.threshold] of the total score.
%
%Arguments:
% CNT - Struct of continuous data
% MRK - Struct of markers
% OPT - Struct or property/value list of optinal parameters, see
%    description above
%
%Returns:
% IVAL - Selected time interval

% Author(s): Benjamin Blankertz
% Carmen Vidaurre, Sep.09


motor_areas= {{'FC5,3','CFC5,3','C5,3','CCP5,3','CP5,3'},
              {'FC1-2','CFC1-2','C1-2','CCP1-2','CP1-2'}, 
              {'FC4,6','CFC4,6','C4,6','CCP4,6','CP4,6'}};
done_laplace= ~isempty(strpatternmatch('* lap*', erd.clab));
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'filter', [], ...
                  'max_ival', [250 5000], ...
                  'start_ival', [750 3500], ...
                  'score_proc', 'r_square_signed', ...
                  'areas', motor_areas, ...
                  'do_laplace', ~done_laplace, ...
                  'ivallength', 500, ...
                  'channelwise', 0);

if isempty(opt.areas),
  opt.areas= {erd.clab};
end

[score_fcn, score_param]= getFuncParam(opt.score_proc);


if opt.channelwise
    if opt.do_laplace,
        erd= proc_channelwise(erd, 'laplace');
    end
    if ~isempty(opt.filter),
        erd= proc_channewise(erd, 'filt', opt.filter.b, opt.filter.a);
    end

    erd= proc_channelwise(erd, 'envelope', 'ma_msec', 200);
    erd= proc_channelwise(erd, 'baseline', [], 'trialwise',0);
else
    if opt.do_laplace,
        erd= proc_laplace(erd);
    end
    if ~isempty(opt.filter),
        erd= proc_filt(erd, opt.filter.b, opt.filter.a);
    end
    
    erd= proc_envelope(erd, 'ma_msec', 200);
    erd= proc_baseline(erd, [], 'trialwise',0);
end

erd= cntToEpo(erd, mrk, opt.max_ival, 'clab', cat(2, opt.areas{:}));
score= feval(['proc_' score_fcn], erd, score_param{:});

%% for each channel calc a positive (class 1 > class 2) and a
%% negative score (class 2 > class 1)
[T, nChans]= size(score.x);
scp= max(0, score.x);
scn= max(0, -score.x);
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

%% determine favorable short time interval
smooth_msec= 250;
smooth_sa= smooth_msec/1000*erd.fs;
timescore= max(scp, scn);
timescore= movingAverage(timescore, smooth_sa, 'centered');
timescore= mean(timescore(:,chansel),2);
tempscore= zeros(size(timescore));
idx= getIvalIndices(opt.start_ival, erd);
tempscore(idx)= timescore(idx);
[topscore,mm]= max(tempscore);
dt= 100/1000*erd.fs;
bidx= [max(1,mm-dt):min(mm+dt,length(timescore))];

%% determine for each channel the favorable sign of class difference
%% and adjust score accordingly
timescore= score.x(:,chansel);
for aa= 1:length(chansel),
  if mean(scp(bidx,chansel(aa))) < mean(scn(bidx,chansel(aa))),
    timescore(:,aa)= -score.x(:,chansel(aa));
  end
end
timescore= mean(timescore, 2);

%% choose initial ival as limited by the two top score time points
[topscore,mm]= max(timescore);
[ss,si]= sort(timescore);
ivalsel= sort(si(end-1:end));

%% iteratively enlarge ival
absscore= max(timescore, 0);
%until here, it is exactly select_timeival.
%in the next, the only difference is:
% the ivallengths are calculated in samples.
ivallength=ceil((opt.ivallength/1000)*erd.fs);
ivallength=ivallength(:);
ivals= zeros(size(ivallength,1),2);
%the ivallengths are order in ascending order
[ivallength ci]=sort(ivallength);
%we do like in select_timeival, only that we finish
%when we reach the longest interval or the maximum length (given as option)
minival=ivalsel(2)-ivalsel(1);
iivals=find(ivallength<minival);
if ~isempty(iivals)
  for iival=iivals
     ivals(iival,1)=ivalsel(1);
     ivals(iival,2)=ivalsel(1)+ivallength(iival);
  end
end;

 while and((ivalsel(2)-ivalsel(1)) < ivallength(end), (ivalsel(2)-ivalsel(1))<(T-1))
  if any((ivalsel(2)-ivalsel(1))==ivallength)
      iival=find((ivalsel(2)-ivalsel(1))==ivallength);
      ivals(iival,:)=ivalsel;
  end;
  b= [ivalsel(1)-1 ivalsel(2)+1];
  if ivalsel(1)==1,
    extend_dir= 2;
  elseif ivalsel(2)==T,
    extend_dir= 1;
  else
    d1= timescore(b);
    d2= [max(timescore(1:b(1))); max(timescore(b(2):end))];
    dirscore= d1+d2;
    [mm,extend_dir]= max(dirscore);
  end
  ivalsel(extend_dir)= b(extend_dir);
 end
  ivals(end,:)=ivalsel;
%end
if any(ivals==0), ivals(ivals==0)=ivalsel(1);,end;
for iival=1:size(ivals,1)
    ivals(iival,:)=erd.t(ivals(iival,:));
end;
ivals=ivals(ci,:);
%ival= erd.t(ivalsel);
