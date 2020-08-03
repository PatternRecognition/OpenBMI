function [y] = shep_sequence(c_pitch,sequence,varargin)


%I. Sturm 1.11.07
%y = shep_sequence(c_pitch,sequence) generates sequence of tones of a Shepard scale with
% center frequency c_pitch if c_pitch is a number, or the pitch name denoted by c_pitch in the two-line octave.
% The pitch of the tones is specified in sequence by the number
% of halftone steps relative to c_pitch. 
% 
%
%[y] = shep_sequence(c_pitch,sequence,'PARAM1',val1, 'PARAM2',val2,...)
%   specifies one or more of the following name/value pairs:
%
%     'durations'    vector specifies the duration of the shepard tones
%							must be same length as sequence
%                    (default 0.2 sec for each tone)
%
%     'fs'   			samplerate
%                    (default 22100 Hz)
%
%     'tones'       	Specifies how many partial tones are used
%							number must be odd to place freq in the center
%							(default: 7)
%
%		'repeats'		number of repetitions of sequence
%
%		'filename'		sequence is saved under 'filename'.wav
%
%examples:
%generates A major cadence (A D E A)
%y=shep_sequence(880,[0 5 7 12]);
%sound(y)
%
%y=shep_sequence(880,[0 4 7 [0 4 7]]);
%generates part of Eb minor scale (Eb F Gb Ab B) twice repeated and saves sequence in'amin.wav'
%y=shep_sequence('Eb',[0 2 3 5 7],'durations',[.1 .1 .1 .1 .4],'tones',7, 'repeats',2,'filename','amin');
%sound(y)
%
%*******************************************
%check number of arguments
error(nargchk(2, 12, nargin));

%defaults
opt=propertylist2struct(varargin{:});
opt=set_defaults(opt,'durations',(ones(size(sequence))*.8),'tones',7,'repeats',1,'fs',22100);

if(length(opt.durations)~=length(sequence))
    error('Vector of durations has not the length as sequence.');
end
               


%check c_pitch:frequency or pitch name?
if(isnumeric(c_pitch))
   freq=c_pitch;

%get center frequency for pitch name
elseif(ischar(c_pitch))
   
   switch(c_pitch)
   case 'C'
      freq=523.25;
   case {'C#','Db'}
      freq=554.37;
   case 'D'
      freq=587.33;
   case {'D#','Eb'}
      freq=622.25;
   case 'E'
      freq=659.26;
   case 'F'
      freq=698.46;
   case {'F#','Gb'}
      freq=739.99;
   case 'G'
      freq=783.99;
   case {'G#','Ab'}
      freq=830.61;
   case 'A'
      freq=880;
   case {'A#','Bb'}
      freq=932.33;
   case 'B'
      freq=987.77;
      
      
   end
 else error('c_pitch has to be a frequency (e.g. 440) or valid pitch (e.g. C#)!')  
end   
   
%generate sequence Shepard-tones
y=[];
for i=1:1:(length(sequence))
   y =[y; shep_tone(freq,opt.tones,sequence(i),opt.durations(i),opt.fs)];
   
end
%repeats
y=repmat(y,opt.repeats,1);


if(isfield(opt,'filename'))
   wavwrite(y,opt.fs,[opt.filename '.wav']); 
end   
