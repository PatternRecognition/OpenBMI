function [y] = shep_chord(c_pitch,tone_matrix,varargin)

%I. Sturm 1.11.07
%y = shep_chord(c_pitch,tone_matrix) generates sequence of chords of tones of a Shepard scale with
% center frequency c_pitch if c_pitch is a number, or the pitch name denoted by c_pitch in the two-line octave.
% The tones that build the chords are specified in the columns of tone_matrix by the number
% of halftone steps relative to c_pitch. Each chord must have the same number of tones.
% Chords with less tones than the maximum number of tones of chords in the sequence have to double tones to match the
% dimension of the maximum chord.
%  
% 
%
%[y] = shep_chord(c_pitch,tone_matrix,'PARAM1',val1, 'PARAM2',val2,...)
%   specifies one or more of the following name/value pairs:
%
%     'durations'     vector specifies the duration of the shepard tones
%                     must be same length as sequence
%                     (default 0.2 sec for each tone)
%
%     'fs'            samplerate
%                     (default 22100 Hz)
%
%     'tones'         Specifies how many partial tones are used
%					  number must be odd to place freq in the center
%					  (default: 7)
%
%	   'repeats'	  number of repetitions of sequence
%
%	   'filename'	  sequence is saved under 'filename'.wav
%      'max_phi'      max. value for phase shift of partial tones against
%                     each other. phase shift is equally distributed.
%                    
%      'sigma'        wideness of gaussian envelope, that specifies the
%                     amplitude of the partial tones
%      'fade'         duration of fade in/fade out time relative to total
%                     duration 
%
%examples:
%generates A major triad broken and as chord
%sound(shep_chord(880,[0 4 7 0;0 4 7 4;0 4 7 7],'fs',22100),22100);
%
%

%
%*******************************************


%defaults
opt=propertylist2struct(varargin{:});
opt=set_defaults(opt,'durations',(ones(size(tone_matrix,2))*.8),'tones',7,'repeats',1,'fs',22100,'max_phi', 0.0, 'sigma', 2, 'fade', 0.08);

if(length(opt.durations)~=size(tone_matrix,2))
    error('Vector of durations has not the length as sequence.');
end
               

%check c_pitch:frequency or pitch name?
if(isnumeric(c_pitch))
   freq=c_pitch;

%get center frequency for pitch name
elseif(ischar(c_pitch))
  freq= get_freq(c_pitch);

 else error('c_pitch has to be a frequency (e.g. 440) or valid pitch (e.g. C#)!')  
end  

 y=[];  
for i=1:size(tone_matrix,2)
    y=[y;shep_tone(freq,tone_matrix(:,i)','tones',opt.tones,'duration',opt.durations(i),'fs',opt.fs,'max_phi',opt.max_phi,'sigma',opt.sigma,'fade',opt.fade,'norm',0)];
   
end
y=y/max(abs(y));
y=repmat(y,opt.repeats,1);


if(isfield(opt,'filename'))
   wavwrite(y,opt.fs,[opt.filename '.wav']); 
end   

