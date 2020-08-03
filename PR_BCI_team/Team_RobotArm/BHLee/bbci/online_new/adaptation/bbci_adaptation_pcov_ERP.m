function [bbci_cls, data_adapt]= bbci_adaptation_pcov_ERP(bbci_cls, data_adapt, marker, varargin)
%BBCI_ADAPTATIOIN_PCOV_ERP - unsupervised adaptation of pooled covariance matrix of an LDA classifier
%
%
%Synopsis:
%  [BBCI_CLS, DATA_ADAPT]= ...
%      bbci_adaptation_pcov_ERP(BBCI_CLS, DATA_ADAPT, 'init', PARAMs, ...)
%  [BBCI_CLS, DATA_ADAPT]= ...
%      bbci_adaptation_pcov_ERP(BBCI_CLS, DATA_ADAPT, MARKER, FEATURE)
%
%This function is called internally by bbci_apply_adaptation.
%
%The selectable parameters are
%  'alpha' - [DOUBLE, default 0.05] Update coefficient for unsupervised
%              adaptation of the pooled covariance matrix
%  'mrk_stimuli'
%            - stimulus marker. This is necesary 
%  'mrk_end_of_segment' 
%            - single marker or list of markers that indicate
%              the end of a data-gathering segment (e.g. a selection in a 
%              speller application). Gathering feature information for the
%              adaptation of one segment ends, when one of those end-markers
%              is received AND at least min_n_data_points (see below) have
%              been collected.
%  'min_n_data_points'
%            - the minimum number of data points (e.g. ERP epochs) that
%            have to be collected before the classifier can be updated

% 06-2012 sven.daehne@tu-berlin.de



if ischar(marker) && strcmp(marker, 'init'),
  opt= propertylist2struct(varargin{:});
  opt= set_defaults(opt, ...
                    'alpha', 0.05, ...
                    'mrk_stimuli', [],...
                    'mrk_end_of_segment', [], ...
                    'min_n_data_points', 50);
  if isempty(opt.mrk_end_of_segment) 
    error('You have to define a marker that signals the end of a data segment  (e.g. trial) and triggers the classifier update!');
  end
  % init the adaptation struct
  data_adapt.opt= opt;
  data_adapt.segment_data = cell(1, 5*opt.min_n_data_points);
  data_adapt.sd_idx = 0;
  data_adapt.last_adaptation_time = -inf;
  data_adapt.last_added_feature_time = -inf;
  if not(isfield(bbci_cls.C, 'cov')) 
      error('bbci_cls.C must have the field "cov" to init the adaptation!');
  end
  if not(isfield(bbci_cls.C, 'mean'))
      error('bbci_cls.C must have the field "mean" to init the adaptation!');
  end
  data_adapt.C_ma = bbci_cls.C.cov;
  data_adapt.mu_ma = bbci_cls.C.mean;
  
  bbci_log_write(data_adapt.log.fid, ...
                 '# %s initialized with the following options: %s.', ...
                 opt.tag, toString(opt));
  return;
else
  feature= varargin{1};
end



% get all events that happened since last adaptation
time= marker.current_time;
check_ival= [data_adapt.last_adaptation_time time];
events= bbci_apply_queryMarker(marker, check_ival);

if ~isempty(events)
    
    n_samples = data_adapt.sd_idx;
    
    % check if mrk_end_of_segment marker is present and if we have enough
    % data for adaptation
    if any(ismember([events(end).desc], data_adapt.opt.mrk_end_of_segment)) && ...
            n_samples >= data_adapt.opt.min_n_data_points
        
        bbci_log_write(data_adapt.log.fid, ...
            '# %s Beginning adapation using %d stored features', ...
            data_adapt.opt.tag, n_samples);
        tic; % start timer
        
        %%% start classifier update/adaptation %%%
        alpha = data_adapt.opt.alpha;
        % compute (regularized) covariance matrix of the segment data
        n_dim = size(data_adapt.C_ma, 1);
        X = zeros(n_dim, n_samples);
        for k=1:n_samples
            X(:,k) = data_adapt.segment_data{k};
        end
        [C_segment, gamma] = clsutil_shrinkage(X);
        
        % update the moving average covariance matrix
        data_adapt.C_ma = (1-alpha)*data_adapt.C_ma + alpha*C_segment;
        
        % update the moving average class mean (TODO for the next version...)
        data_adapt.mu_ma = data_adapt.mu_ma;
                
        % update the classifier
        mu_diff = data_adapt.mu_ma(:,2)-data_adapt.mu_ma(:,1);
        mu_sum = sum(data_adapt.mu_ma, 2);
        C_inv = pinv(data_adapt.C_ma);
        w = C_inv * mu_diff;
        b = -0.5 * w'*mu_sum;
        bbci_cls.C.w = w;
        bbci_cls.C.b = b;
        bbci_cls.C.cov = data_adapt.C_ma;
        bbci_cls.C.mean = data_adapt.mu_ma;
        %%% end classifier update/adaptation %%%
                
        % clear the data list and begin new interval for event listening
        for k=1:length(data_adapt.segment_data)
            data_adapt.segment_data{k} = [];
        end
        data_adapt.sd_idx = 0;
        data_adapt.last_adaptation_time= time;
        
        bbci_log_write(data_adapt.log.fid, ...
            '# %s Classifier retraining done! Duration: %g seconds', ...
            data_adapt.opt.tag, toc);
        
    end
    
    % if there is a new data point, check if it is an ERP stimulus and
    % store it if yes
    if not(isempty(feature.x)) 
        % find the marker that corresponds to this feature
        idx = [];
        for k=1:length(events)
            if events(k).time == feature.time
                idx = k;
            end
        end
        
        if not(isempty(idx)) && ...
                ismember(events(idx).desc, data_adapt.opt.mrk_stimuli) && ...
                not(data_adapt.last_added_feature_time == feature.time)
            
            if data_adapt.sd_idx > length(data_adapt.segment_data)
                bbci_log_write(data_adapt.log.fid, ...
                '# %s Allocating more space to save features!', data_adapt.opt.tag);
            
                bigger_segment_data = cell(1, 2*length(data_adapt.segment_data));
                bigger_segment_data(1,1:length(data_adapt.segment_data)) = ...
                    data_adapt.segment_data;
                data_adapt.segment_data = bigger_segment_data;
            end
            bbci_log_write(data_adapt.log.fid, ...
                '# %s Storing feature with marker = %d', data_adapt.opt.tag, events(idx).desc);
            data_adapt.sd_idx = data_adapt.sd_idx + 1;
            data_adapt.segment_data{data_adapt.sd_idx} = feature.x;
            data_adapt.last_added_feature_time = feature.time;
        end
    end
end

