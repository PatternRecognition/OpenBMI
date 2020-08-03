function [bbci, data]= bbci_calibrate_NIRS(bbci, data)
%BBCI_CALIBRATE_NIRS - Calibrate for ERP-design with NIRS
%
%This function is called by bbci_calibrate 
%(if BBCI.calibate.analyze_fcn is set to @bbci_calibrate_NIRS_ERP).
%Via BBCI.calibrate.settings, the details can be specified, se below.
%
%Synopsis:
% [BBCI, DATA]= bbci_calibrate_NIRS_ERP(BBCI, DATA)
% 
%Arguments:
%  BBCI -  the field 'calibrate.settings' holds parameters specific to
%          calibrate CSP-based BCI processing.
%  DATA -  holds the calibration data
%  
%Output:
%  BBCI - Updated BBCI structure in which all necessary fields for
%     online operation are set, see bbci_apply_structures.
%  DATA - As input but added some information of the analysis that might
%     be reused in a second run
%
%BBCI.calibrate.settings may include the following parameters:
%  visu_clab:   Clab used for visualization
%  lp:          butterworth lowpass filter frequency
%  derivative   use derivative (slope) of signal as feature
%  classes: [1x2 CELL of CHAR] Names of the two classes, which are to be
%     discriminated. For classes = 'auto', all pairs of available classes
%     are investigated, and the one with best xvalidation performance is
%     chosen. Default is 'auto'.   **** TO BE IMPLEMENTED ****
% Display options:
%  zscore:      (for display purposes) channels are normalized to zero mean
%               and unit variance 
%  grid:        provide a string specifying the grid layout for grid_plot
%
% ------------
%  ival: [1x2 DOUBLE] interval on which CSP is performed, 'auto' means
%     automatic selection. Default is 'auto'.
%  band: [1x2 DOUBLE] frequency band on which CSP is performed, 'auto' means
%     automatic selection. Default is 'auto'.
%  clab: [CELL] Labels of the channels that are used for classification,
%     default {'*'}.
%  nPatters: [INT>0] number of CSP patterns which are considered from each
%     side of the eigenvalue spectrum. Note, that not neccessarily all of
%     these are used for classification, see settings.pattern.
%     Default is 3.
%  patterns: [1xN DOUBLE or 'auto'] vector specifying the indices of the
%     CSP filters that should be used for classification; 'auto' means
%     automatic selection. Default is 'auto'.
%  model: [CHAR or CELL] Classification model.
%     Default {'RLDAshrink', 'gamma',0, store_means',1, 'scaling',1}.
%  reject_artifacts:
%  reject_channels:
%  reject_artifacts_opts: cell array of options which are passed to 
%     reject_varEventsAndChannels.
%  check_ival: interval which is checked for artifacts
%  do_laplace: do Laplace spatial filtering for automatic selection
%     of ival/band. If opt.do_laplace is set to 0, the default value of
%     will also be set to opt.visu_laplace 0 (but can be overwritten).
%  visu_laplace: do Laplace filtering for grid plots of Spectra/ERDs.
%     If visu_laplace is a string, it is passed to proc_laplace. This
%     can be used to use alternative geometries, like 'vertical'.
%  visu_band: Frequency range for which spectrum is shown.
%  visu_ival: Time interval for which ERD/ERS curves are shown.
%  grd: grid to be used in the grid plots of spectra and ERDs.
%
%Only the figures of the chosen class combination are visible, while the
%others are hidden. In order to make them visible, type
%>> set(cat(2, data.all_results.figure_handles), 'Visible','on')
%
%You might like to modify bbci.feature.ival after running this function.


calibrate = bbci.calibrate;
opt= calibrate.settings;
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'classes', 'auto', ...
                 'clab', {'*'}, ...
                 'classifier','RLDAshrink',...
                 'ref_ival',[-2000 0], ...
                 'ival',[-2000,10000], ...
                 'nIvals', 5, ...
                 'signal','both', ...
                 'doLowpass', 1, ...
                 'derivative', 0, ...
                 'baseline',1, ...
                 'zscore',1, ...
                 'grid',[] , ...
                 'spectra', 1, ...
                 'plot',1 ...
                 );


opt_grid = {'scaleGroup' 'all'};
opt_tit = {'Fontsize' 10 'Fontweight' 'bold'};

%% Build dummy grid for grid_plot
clab = strhead(data.cnt.clab(1:end/2));
clab = {'scale',clab{:},'legend'};
ii=1;
grd = '';
for r=1:ceil(sqrt(numel(clab)-2)) % nr of rows
  for c=1:ceil(sqrt(numel(clab)-2))
    if ii<=numel(clab)
      grd = [grd clab{ii} ','];
    else
      grd = [grd '_,'];
    end
    ii=ii+1;
  end
  grd= [grd(1:end-1) '\n']; 
end
grd = sprintf(grd);
mnt= mnt_setGrid(data.mnt, grd);

%% *** Preprocessing ***
if ~isempty(opt.clab)
  data.cnt= proc_selectChannels(data.cnt,opt.clab);
end

% LP filter data [IS IT POSSIBLE TO SET BBCI.SIGNAL AND DO THIS
% AUTOMATICALLY?]
if opt.doLowpass
  fprintf('Lowpass filtering signal\n')
  [filt_b,filt_a]=butter(3, opt.lp*2/data.cnt.fs,'low');
  data.cnt = proc_filtfilt(data.cnt,filt_b,filt_a);
end

% oxy = dat;
% oxy.x = oxy.x(:,1:end/2);
% oxy.clab = oxy.clab(:,1:end/2);
% deoxy = dat;
% deoxy.x = dat.x(:,end/2+1:end);
% deoxy.clab = deoxy.clab(:,end/2+1:end);

if opt.spectra
  % Spectra
  spec_band = [0.01 3];     % Frequency band for spectrum plot
  spec_win = kaiser(250,2);   % window for proc_spectrumspecoxy = proc_spectrum(oxy, spec_band,spec_win);
  spec = proc_spectrum(data.cnt, spec_band,spec_win);
end

% derivative feat?
if opt.derivative
   fprintf('Calculating derivative of NIRS signal\n')

   data.cnt.x=diff(data.cnt.x);
%    deoxy.x=diff(deoxy.x); %*data.cnt.fs;
%    oxy.x=diff(oxy.x); 
end

%% Epoch data
data.cnt = cntToEpo(data.cnt,data.mrk,opt.ival);
    
% baseline
if opt.baseline   
  fprintf('Performing baseline correction\n')
  data.cnt = proc_baseline(data.cnt,opt.ref_ival);
end

% oxy.clab = untex(oxy.clab);
% deoxy.clab = untex(deoxy.clab);
% 
%% *** Neurophysiology ***
% Split in oxy/deoxy or WL1/WL2
% oxy = data.cnt;
% oxy.x = oxy.x(:,1:end/2,:);
% oxy.clab = oxy.clab(:,1:end/2);
% deoxy = data.cnt;
% deoxy.x = data.cnt.x(:,end/2+1:end,:);
% deoxy.clab = deoxy.clab(:,end/2+1:end);

opt_tit = {'Fontsize' 10 'Fontweight' 'bold'};

%% make plots
if opt.plot
    %% Power spectrum of CNT data
    if opt.spectra
      fig_set(2),clf
      subplot(2,1,1)
      plot(spec.t,spec.x(:,1:end/2)), title(strtail(spec.clab{1}),opt_tit{:})
      xlabel(sprintf('Frequency [%s]',spec.xUnit)),ylabel(sprintf('Power [%s]',spec.yUnit))
      subplot(2,1,2)
      plot(spec.t,spec.x(:,1:end/2)), title(strtail(spec.clab{1}),opt_tit{:})
      xlabel(sprintf('Frequency [%s]',spec.xUnit)),ylabel(sprintf('Power [%s]',spec.yUnit))
      annotation('textbox',[.01 .9 .1 .1 ],'String','Power spectra of CNT data ');
    end

    %% z-score normalization 
    if opt.zscore
      fprintf('Calculating z-scores\n')
      m= mean(data.cnt.x,1);   % mean
      s= std(data.cnt.x,1);    % std
      len_x = length(data.cnt.x);
      data.cnt.x= (data.cnt.x - repmat(m,[len_x 1])) ./ repmat(s,[len_x 1]);
      clear m s 
    end

    %% Grid plot -- signal
    fig_set(1),clf
    H = grid_plot(proc_selectChannels(data.cnt,1:numel(data.cnt.clab)/2), mnt, defopt_scalp_power,opt_grid{:});
    set(gcf,'Name',['Epoched signal ' strtail(data.cnt.clab{1})])
    % printFigure(['grid_oxy' saveapp], [36 20], opt_fig);  

    fig_set(2),clf
    H = grid_plot(proc_selectChannels(data.cnt,(numel(data.cnt.clab)/2)+1:numel(data.cnt.clab)), mnt, defopt_scalp_power,opt_grid{:});
    set(gcf,'Name',['Epoched signal ' strtail(data.cnt.clab{end})])
    % printFigure(['grid_deoxy' saveapp], [36 20], opt_fig);

    %% R-Square plots
    dat_r = proc_r_square_signed(data.cnt,'multiclass_policy','pairwise');
    npairs = size(dat_r.y,1);
    nclab = numel(data.cnt.clab)/2;
    for ii=1:npairs
    for od=[1 2]   %oxy-deoxy
      figure
      imagesc(dat_r.x(:,(od-1)*nclab+1:od*nclab,ii)')
      title([dat_r.className{ii} ' ' strtail(dat_r.clab{od*nclab})],opt_tit{:})
      ylabel('Channel',opt_tit{:}),xlabel('Time',opt_tit{:})
      xt=  str2num(get(gca,'XTickLabel'));
    %   yt=  str2num(get(gca,'YTickLabel'));
      set(gca,'XTickLabel',dat_r.t(xt))
    %   set(gca,'YTickLabel',strhead(oxy_r.clab))
      colorbar
    end
    end
    clear dat_r

    %% Wavelets
    spec = proc_selectChannels(data.cnt,1:nclab);
    spec = proc_specgram(spec, [0.01 3]);

    % data.cnt = proc_selectChannels(data.cnt,1:nclab);
    % wave_band = [.01:.4:3];
    % for w=wave_band
    %   if w==wave_band(1)
    %     wave = proc_wavelets(data.cnt,w,'vectorize',1);
    %   else
    %     w1 = proc_wavelets(data.cnt,w,'vectorize',1);
    %   end
    % end
end
%% *** Classification ***
% BBCI.FEATURE selection
bbci.feature.fcn = {@proc_jumpingMeans};

% class combination
if isequal(opt.classes, 'auto'),
  class_combination= nchoosek(1:size(data.mrk.y,1), 2);
else
  class_combination= find(ismember(data.mrk_all.className, opt.classes));
  if length(class_combination) < length(opt.classes),
    error('Not all specified classes were found.');
  end
  if length(class_combination) ~= 2,
    error('This calibration is only for binary classification.');
  end
end

ival_cfy={};

for ci= 1:size(class_combination,1)
  
    fv=proc_selectClasses(data.cnt,class_combination(ci,:));
    fprintf('\nComparing classes %s vs %s.',fv.className{1},fv.className{2})
  
    % find the feature/time intervals
    fvr = proc_r_square_signed(fv);
    [ival_cfy{ci},nfo]= select_time_intervals(fvr,'nIvals',opt.nIvals,'clab_pick_peak', fv.clab,...
      'visualize',0,'intersample_timing',1);

    bbci.feature.param = { {ival_cfy{ci}} };
    % Get feature vector ans train classifier
    fv = bbci_calibrate_evalFeature(fv, bbci.feature);

    bbci.classifier.C= trainClassifier(fv, opt.classifier);

    
%     % Xvalidation
%     [loss, loss_std]= xvalidation(fv, opt.classifier, ...
%        'sample_fcn', {'divisions', [opt.nShuffles opt.nFolds]},'save_classifier',0);
% 
%     mean_loss(ci)= loss;
%     std_loss(ci)= loss_std;
    
    cfy_fields= {'signal', 'feature', 'classifier', 'quit_condition'};
    bbci_all(ci)= copy_struct(bbci, cfy_fields);

    clear fv*
end

%% Choose best binary combination of classes
% Copy parameters into BBCI struct

nComb= size(class_combination,1);
if nComb > 1,
%   [dummy, bi]= min(mean_loss + 0.1*std_loss);
bi=2;
  bbci= copy_fields(bbci, bbci_all(bi), cfy_fields);
%     bbci.feature.param = { {ival_cfy{ci}} };
end

comb = class_combination(bi,:);
fprintf('Chosen (best) combination: classes %s vs %s.\n',data.cnt.className{comb(1)},data.cnt.className{comb(2)})
bbci.classifier.classes=data.cnt.className(comb);
%% Set the BBCI struct

% BBCI.SIGNAL preprocessing (really just for online data?)
% bbci.signal.clab = opt.clab; 
bbci.signal.clab = data.cnt.clab;

% data.signal.clab = opt.clab;
bbci.signal.proc = {};
% if ~isempty(opt.clab)
%   bbci.signal.proc = {bbci.signal.proc{:} {@proc_selectChannels,opt.clab}};
% end
if opt.doLowpass
  bbci.signal.proc = {bbci.signal.proc{:} {@proc_filtfilt,filt_b,filt_a}};
end
if opt.derivative
  bbci.signal.proc = {bbci.signal.proc{:} {@diff}};
end

% SOMEWHERE: proc_baseline ?

% BBCI.FEATURE selection
bbci.feature.ival = opt.ival;
