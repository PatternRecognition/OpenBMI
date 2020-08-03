function erpc = erp_components(epo,comp,ival,clab,varargin)

% ERP_COMPONENTS 
% Finds an ERP component (local maximum or minimum) within designated
% intervals. Prefix of the component (N or P) is used to determine whether
% a positive or negative peak is searched. 
%
% Synopsis:
%   ERP = ERP_COMPONENTS(EPO,COMP,IVAL,CLAB,<OPT>)
%
% Arguments:
%   EPO:        a struct with the epoched data.
%   COMP:       name of the component (e.g., N1), 'P' or 'N' is enough
%   IVAL:       corresponding interval in ms
%   CLAB:       cell array of channel labels 
%
% OPTIONS:
%      class      - specify for which class number ERP is searched (default
%                   1)
%      extremum   - if 1 takes most extreme value in an interval if a peak
%                   is not found. If 0, the mean is taken (default)
%
% OUT: ERP  - array of erp structs (one struct for each experimental file/subcondition),
% specifying the peak amplitude and latency for each component. 
% New fields:
% .amplitude - for each electrode, the negative or positive peak
%                   amplitude
% .latency   - for each electrode, the  corresponding latency
%
% Example:
% erp = erp_components(epo,'n',[200 300],'Cz');
% finds the most negative peak in the interval 200-300 for channel Cz.
%
% 2009 Matthias Treder

warning off
%% Opt parameters
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
              'class', 1,...
              'extremum',0 ...
              );

if ~iscell(clab)
  clab = {clab};
end

%% Calculate ERPs, and select channels & classes of interest
if ~(size(epo.x,3)==numel(epo.className))
  epo = proc_average(epo);
end

erpc = struct();
erpc.amplitude = [];
erpc.latency = [];

epo = proc_selectChannels(epo,clab);
epo = proc_selectClasses(epo,opt.class);

%% Find peak amplitudes and latencies

% For each electrode
negative = strcmpi(comp(1),'n');  % Check whether negative or positive component
tx = find( epo.t>=ival(1) & epo.t<=ival(2) );       % Indices of x values corresponding to time window
for e=1:numel(clab)      % Traverse electrodes
    % Find (indices of) all peaks within interval
    peakx = []; peaky = [];
    for ii = 1:numel(tx)
        cx = tx(ii);          % Current index
        val = epo.x(cx,e);
        valmin1 = epo.x(cx-1,e);
        valplus1 = epo.x(cx+1,e);
        if (negative && val<valmin1 && val<valplus1)...
                || (~negative && val>valmin1 && val>valplus1)
            peakx = [peakx cx];
            peaky = [peaky val];
        end
    end
    % Find peak and save amplitude + latency
    if negative
        [extremum ind] = min(peaky);
    else [extremum ind] = max(peaky);
    end
    if isempty(extremum)
      if opt.extremum
        fprintf('No peak found at electrode %s for class %s, taking absolute extremum\n', ...
          clab{e},num2str(epo.className{1}));
        if negative; [extremum ind] = min(epo.x(tx,e)); 
        else [extremum ind] = max(epo.x(tx,e)); 
        end;
        erpc.latency(e) = epo.t(tx(ind));
      else
        fprintf('No peak found at electrode %s for class %s, taking mean\n', ...
          clab{e},num2str(epo.className{1}));
        extremum = mean(epo.x(tx,e)); 
        erpc.latency(e) = (tx(1)+tx(end))/2;
      end
    else
        erpc.latency(e) = epo.t(peakx(ind));
    end;
    erpc.amplitude(e) = extremum;
end

end