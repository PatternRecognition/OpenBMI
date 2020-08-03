function mrk= mrk_chronSplit(mrk, varargin)

% MRK_CHRONSPLIT - chronologically splits the markers into even-sized
% classes, such as an 'early' class (containing the first 50% of events)
% and a 'late' class (containing the last 50% of events).
%
% Synopsis:
%   [MRK]= mrk_chronSplit(mrk, <nClasses>, <appendix>)
%
% Arguments:
%   MRK: marker structure
%
% Optional arguments:
%   NCLASSES: number of classes in which the dataset is to be split
%   (default 2)
%   APPENDIX: cell array with labels for the classes (default {'early' 'late'})
%
% Returns:
%   MRK: updated marker structure
%
% Author: Benjamin B
% 7-2010: Documented, cleaned up (Matthias T)


%% Process input 
if nargin == 1
  nClasses = 2;
  appendix={'early', 'late'};
elseif nargin == 2
  nClasses = varargin{1};
  switch(nClasses)
    case 2; appendix= {'early', 'late'};
    case 3; appendix= {'early', 'middle', 'late'}; 
    otherwise,
      for ii=1:nClasses
        appendix{ii} = ['block' num2str(ii)];
      end
  end
else
  nClasses = varargin{1};
  appendix = varargin{2};
end

%% 
className= mrk.className{1};

nEvents= length(mrk.pos);
inter= round(linspace(0, nEvents, nClasses+1));

mrk.y= zeros(nClasses, nEvents);
mrk.className= cell(1, nClasses);
for ic= 1:nClasses,
  idx= inter(ic)+1:inter(ic+1);
  mrk.y(ic,idx)= 1;
  mrk.className{ic}= [className ' (' appendix{ic} ')'];
end
