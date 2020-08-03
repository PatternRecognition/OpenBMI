function [arti,Sources] = get_ArtifactsPCA(cnt,artifact, artifactmarkers, ...
				     options, varargin)
% cnt is the real EEG
% artifact are the EEG of the artifact measurement
% artifactmarkers are the markes in the artifact measurement
% options is a struct (or for the field 'same' only as a number) with
% the following entries:
%        same: a flag which says if you want for each artifact
%        their own SourceVectors (0) or SourceVectors for the
%        covariances of all together.
%        channels: a cell array with channels in the usual format
%        for make_Segments, where the algorithm only makes PCA. YOu
%        get then only this channels as ocular potentials back
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
% Output: arti are the expected EEG getting from ocular
% potentials. To get the real EEG (without ocular potentials you
% must subtract this of the EEG)
% Sources are a struct with the SOurces Vectors, SourceWaveforms,
% values(see get_SourcesPCA.m)
%
% Guido Dornhege
% 28.03.02

if nargin<5 
  error('not enough input arguments')
end


same = 0;
if exist('options') & ~isempty(options) 
  if ~isstruct(options) & options == 1
    same = 1;
  end
  if isstruct(options) & isfield(options,'same') & options.same == 1
    same = 1;
  end
end

if nargout>1
  [SourceVectors, SourceWaveforms, values] = ...
      get_SourcesPCA(artifact,artifactmarkers,options, varargin{: ...
		   });
  Sources.SourceVectors = SourceVectors;
  Sources.SourceWaveforms = SourceWaveforms;
  Sources.values = values;
else
  SourceVectors = get_SourcesPCA(artifact,artifactmarkers,options, ...
					  varargin{:});
end

if same==0
  SourceVectors = orth(SourceVectors);
end

if exist('options') & isstruct(options) & isfield(options,'channels') ...
      & ~isempty(options.channels)
  cnt = proc_selectChannels(cnt,options.channels{:});
end
  
arti = copyStruct(cnt,'x');
arti.x = cnt.x*SourceVectors*SourceVectors';

















