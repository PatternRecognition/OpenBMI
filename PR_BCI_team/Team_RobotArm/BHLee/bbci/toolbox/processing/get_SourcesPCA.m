
function [SourceVectors, SourceWaveforms, values] = get_SourcesPCA(cnt,mrk,options, varargin)
%GET_SOURCESPCA gets some number of sources with PCA
%
% Input: cnt, mrk usual data structure you get from loadProcessedEEG file. 
% options is a struct (or for the field 'same' only a number) with
% the following entries:
%        same: a flag which says if you want for each artifact
%        their own SourceVectors (0) or SourceVectors for the
%        covariances of all together.
%        channels: a cell array with channels in the usual format
%        for make_Segments, where the algorithm only makes PCA. YOu
%        get then only this channels as Sources Vectors back
%        (default all)
%
% varargin are the artifacts.
% One artifact is a struct with the following entries (in braces
% the defaults, if missing)
%        - name: a name for artifact 
%        - artifacts: a cell array with artifacts or a string with
%          one artifact( name)
%        - readSamples: a method for reading samples out of the the
%          cnt ('readSamples_simple'), the program must give back a
%          3-dim matrice (rows: time, column: channel, z: sample)
%        - readOptions: maybe some options for the call of
%          readsamples ([])
%        - number: number of SourceVectors you want to get back (1)
%
% Output:
% SourceVectors: a matrice where in each column are one
% SourceVector
% SourceWaveforms: a cell array with all SOurceWaveforms
% values: struct with
%      eigenvalues: the eigenvalues in PCA
%      names: a cell array with the names and numbers (the
%      eigenvector to the biggest eigenvalue).
%      proportions: the proportion of Eigenvalues on the whole matrice
%
% Guido Dornhege
% 27.03.02

% check inputs
if nargin<4 
  error('not enough input arguments'); 
end
if ~exist('options') | isempty(options)  
  option.same = 0;
end
if ~isstruct(options)
  options.same = options;
end
if ~isfield(options,'same')
  options.same = 0;
end
if ~isfield(options,'channels')
  options.channels = [];
end


if ~isempty(options.channels) & ~iscell(options.channels)
  options.channels = {options.channels};
end
  
% for each artifact do something
for arti = 1:length(varargin)
  artifact = varargin{arti};

  % see first for defaults
  if ~isstruct('artifact')
    artifact.name = artifact;
  end
  
  if ~isfield(artifact,'name')
    if ~isfield(artifact,'artifacts')
      error('Artifact not specified');
    end
    if iscell(artifact.artifacts)
      artifact.name = artifact.artifacts{1};
    else
      artifact.name = artifact.artifacts;
    end
  end
  
  if ~isfield(artifact,'artifacts')
    artifact.artifacts = artifact.name;
  end
  
  if ~iscell(artifact.artifacts)
    artifact.artifacts = {artifact.artifacts};
  end
  
  if ~isfield(artifact, 'readSamples')
    artifact.readSamples = 'readSamples_simple';
  end
  
  if ~exist(artifact.readSamples,'file')
    error('file does not exist');
  end
  
  if ~isfield(artifact,'readOptions')
    artifact.readOptions = {[]};
  end
  
  if ~iscell(artifact.readOptions)
    artifact.readOptions = {artifact.readOptions};
  end
  
  if ~isfield(artifact,'number')
    artifact.number = 1;
  end
  
  
  % now get the intervals
  iv = get_relevantArtifacts(mrk, artifact.artifacts{:});
  
  % now get the samples
    
  samp = feval(artifact.readSamples, cnt, iv, ...
	       artifact.readOptions{:});
  if ~isempty(options.channels)
    samp2 = copyStruct(cnt,'x');
    samp2.x = samp;
    samp = getfield(proc_selectChannels(samp2,options.channels{:}), ...
		    'x');
    clear samp2
  end
  
  % mean of the samples
  
  meansamp{arti} = mean(samp,3);
  
  % get the covs of the samples
  %covsamp{arti} = cov(meansamp{arti},1);
  covsamp{arti} = transpose(meansamp{arti})*meansamp{arti}/size(meansamp{arti},1);
  
  if options.same
    numb(arti) = size(samp,2);
    artinumb(arti) = artifact.number;
  end
  
  
  % get the eigenvalues
  [SV{arti}, D{arti}]  = eigs(covsamp{arti}, artifact.number,'LM', ...
			      struct('disp',0));
  
  SV{arti} = SV{arti}*diag(2*((max(SV{arti})+min(SV{arti}))>0)-1);
  
      
end

% Now prepare the end

if options.same
  covall = zeros(size(covsamp{1},1));
  for i = 1:length(covsamp)
    covall = covall + covsamp*numb(i);
  end
  covall = covall/sum(numb);
  
  [SourceVectors, D] = eigs(covall, sum(artinumb), 'LM', ...
			    struct('disp',0));
  if nargout>1
    for i =1 : length(meansamp)
      SourceWaveforms{i} = {pinv(SV{i}'*SourceVectors)* ...
	  transpose(meansamp{i})};
    end
    if nargout>2
      values.eigenvalues = diag(D);
      values.proportions = sum(values.eigenvalues)/trace(covall);
    end
  end
else
  SourceVectors = cat(2,SV{:});
  if nargout>1
    for i =1 : length(meansamp)
      SourceWaveforms{i} = pinv(SV{i})* transpose(meansamp{i});
    end    
    if nargout>2
      values.eigenvalues = [];
      for k = 1:length(D)
	values.eigenvalues = cat(2,values.eigenvalues,diag(D{k}));
	values.proportions(k) = sum(diag(D{k}))/trace(covsamp{k});
      end      
    end
  end
end

if nargout>2
  values.names = [];
  for i = 1:length(meansamp)
    na = varargin{i}.name;
    if varargin{i}.number <2
      values.names = cat(2,values.names,{na});
    else
      for j = 1:varargin{i}.number
	values.names = cat(2,values.names,{[na, ' ', ...
		    num2str(j)]});
      end
    end
  end  
end

        

if nargout >2
    SV = [];
    for i = 1:length(SourceWaveforms)
        for j = 1:size(SourceWaveforms{i},1)
            SV = cat(2,SV, {getfield(SourceWaveforms{i},{j,1:size(SourceWaveforms{i},2)})});
        end
    end
    SourceWaveforms = SV;
end
















