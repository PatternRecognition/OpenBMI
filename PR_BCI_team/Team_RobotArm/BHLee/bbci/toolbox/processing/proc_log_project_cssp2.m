% fv = proc_log_project_cssp2(fv, W, F, opt)
% WITH LOG

% ryotat

function fv = proc_log_project_cssp2(fv, W, F, varargin)
[T,d,n]=size(fv.x);


opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, struct('spec', 1, 'nWin', 1));
nfft = size(F,1);
if nfft<T
  warning('PROC_LOG_PROJECT_CSSP2:winOvlp', 'nfft<T. Overlapping windows used.');
  opt.nWin=[];
end

fv = proc_linearDerivation(fv, W);

if opt.nWin==1
  fv.x = abs(fft(fv.x)).^2;
else
    %% MSK: something wrong here, wrong pairing of opts for proc_CSD
  %fv = proc_CSD(fv, nfft, opt);
  fv = proc_CSD(fv, opt);
  
end

fv.x = log(dot(repmat(F, [1,1,n]), fv.x, 1));
