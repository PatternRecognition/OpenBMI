function qa_plotPsychometricFktn(R, varargin)
%
% USAGE:       function qa_plotPsychometricFktn(R)
%
% Plots the psychometric data and fits a psychometric function to the data.
%
% IN:       R           -       output of qa_getDetectionData  
%                               OR: epo or marker structure with field
%                               .detected
%
% NOTE: This function is dependent on the psignifit toolbox for matlab (mpsignifit)
%       (cf. psignifit/sourceforge.net)
%
% Simon Scholler, June 2011
%

opt= propertylist2struct(varargin{:});
opt = set_defaults(opt, ...
                 'nafc', 1, ... % 1: yes-no task; 2: 2-alternative forced choice
                 'method', 'bootstrap', ...
                 'xlabel', 'Stimulus Level', ...
                 'ylabel', 'Detection Rate', ...
                 'color', [0 0 1]); % bootstrap or bayes  

% cd to psignifit toolbox directory
global BCI_DIR
d = pwd;
cd([BCI_DIR '/import/psignifit3/mpsignifit/'])

if isstruct(R)  % if epo or marker struct
    R = qa_getDetectionData(R);
end
R(:,2)= R(:,2).*R(:,3);  % convert hit-percentages to #hits

priors.m_or_a = 'None';
priors.w_or_b = 'None';
priors.lambda = 'Uniform(0,.03)';
priors.gamma  = 'Uniform(0,.1)';
switch opt.method
    case 'bayes'
        psyfct = BayesInference ( R, priors, 'nafc', 1, 'samples', 200);
    case 'bootstrap'
        psyfct = BootstrapInference ( R, priors, 'nafc', 1, 'samples', 200);
    otherwise
        error('Method string unknwon')
end
plotPMF(psyfct, 'ylabel', opt.ylabel, 'xlabel', opt.xlabel, 'color', opt.color);
cd (d)


