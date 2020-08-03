function [order, sounds_key] = load_mimu(varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Irene Sturm 11/2007
% [order, sounds_key] = load_mimu() generates sequence of keys for MiMu-experiments and loads
%                       the .wav-files needed.
% out:
%
%     'order'       struct with fields:
%                   .probe_tones    : vector that specifies the interval of each pitch change in number of
%                   semitones. E.g. [1 4 7] means the first change of pitch is one semitone up, the second 
%                   4 semitones up.
%                   .sequence       : sequence of keys played.
%                   .reps           : number of repetitions for each trial.
%     'sounds_key'  cell array (size [1 12]) containing the sounds
%                   for 12 pitches used.
%
% 
%
% [order, sounds_key]=load_mimu('PARAM1',val1, 'PARAM2',val2,...)
%   specifies one or more of the following name/value pairs:
%
%
%     
%     'stim_type'       'shepard' or 'piano' (synthetic)
%     'probe_tones'     vector of numbers between 1 and 11. Specifies which intervals of pitch changes are used.
%                       default [1:11] (all intervals)
%     'howmany'         how many sequences are played (for testing)
%
%     'samples_per_class' Number of trials for each class. Total number of
%                           trials is 'samples_per_class'*number of
%                           probe_tones.Default 10.
%     'fade'            duration of fade in/fade out time relative to total
%                       duration 
%     'minrep'/'maxrep' minimal/maximal number of repetitions for keys. Default 6 and 11.
%     'duration'        Duration of each tone. Default 0.38 s
%     'aso'             Asynchronous stimulus onset.Time between one onset
%                       and the next.Default 0.42.
%     'save'            save struct order to .mat-file as specified by
%                       opt.file_name
%     'fs'             sampling rate 
%     'block'          number of experiment. For several runs opt.block has
%                      to be incremented for each.
% examples: 
% [order, sounds_key] = load_mimu()
% probe_tone_exp(order, sounds_key)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         

%STIM_DIR='C:\users\irene\BCI\matlab\svn\bbci\investigation\personal\irene\music_cognition\muscog2\Sounds\';

opt=struct;
opt = propertylist2struct (varargin{:});
opt = set_defaults (opt, 'stim_type','shepard','probe_tones',[1:11],'samples_per_class',10,'min_rep',6,'max_rep',11,'duration',0.38,'aso',0.44,'save',0,  'fs',44100, 'fade', 0.08,'major_or_minor','major','block',1);
global STIM_DIR
order = struct;
pitches = {'C','C#','D','Eb','E','F','F#','G','G#','A','Bb','B'};
probe_tones=opt.probe_tones;
num_samples_per_class=opt.samples_per_class;
numTrials=num_samples_per_class*size(probe_tones,2);

%initialize matrix of stim combinations
if (~isfield(opt,'predefined_probe_tones'))
A = zeros(1,numTrials);
A_temp=repmat(probe_tones,1,num_samples_per_class);
rp=randperm(numTrials);
A=A_temp(rp);
order.probe_tones=A;
else
    order.probe_tones=opt.predefined_probe_tones;
end
%initialize starting point randomly
rp = randperm(11);
st_key = pitches{rp(1)};
order.sequence{1}=st_key;
old_key=rp(1);
%generate sequence of keys according to probe tone vector
for i=1:size(order.probe_tones,2)
    new_key=old_key+order.probe_tones(i);
    if new_key>12
        new_key=new_key-12;
    end
    order.sequence{i+1}=pitches{new_key};
    old_key=new_key;
end



if(opt.save)
    fn=[opt.filename '_' num2str(opt.block)];
    save(fn,'order')
    fprintf(['order saved to ' fn '.mat\n']);
end
nSamples=opt.fs*opt.duration;
num_trials = size(order.sequence,2);

%generate random order of equally distributed repetition levels
rep_levels= opt.min_rep:opt.max_rep;
n = floor(num_trials/size(rep_levels,2));
reps = [];
for k = 1:n
    reps=[reps rep_levels];
end
reps=[reps rep_levels(1:mod(num_trials,size(rep_levels,2)))];
rp_reps=randperm(size(reps,2));
reps=reps(rp_reps);
order.reps=reps;
min=Inf;

if (~isempty(strmatch(opt.stim_type,'shepard')))
   stim_type='Shepard\test_shepard_'; 
elseif (strmatch(opt.stim_type,'piano'))
    stim_type='Piano_synthetic\test_piano_';
end 

%load sounds
for k = 1:size(pitches,2)
    fprintf('loading pitch %d \n',k)
   
    key_file=[STIM_DIR stim_type pitches{k} '_' opt.major_or_minor];
    
    
    [y1,fs,nbits]=wavread(key_file); 
    
    if size(y1,1) < min
        min=size(y1,1);
    end
    if (strmatch(opt.stim_type,'shepard'))
    y1=[y1 y1]; 
end
 
    sounds_key{k} = y1; 
    
end
%cut length of stimuli to opt.duration, apply amplitude envelope
for i=1:size(sounds_key,2)
    y1=sounds_key{i};
    if(nSamples<=min)
    y1=y1(1:nSamples,:);
    else
    y1=y1(1:min,:);
    fprintf('Duration too long!\n')
    end
    % fade in/out
    len = size(y1, 1);
	uebergang = round (len * opt.fade);
	z = 1:uebergang;
	z = z / uebergang;
    z=z';
	y1(1:uebergang,:) = y1(1:uebergang,:) .* [z z];
	y1(end-uebergang+1:end,:) = y1(end-uebergang+1:end,:) .* [flipud(z) flipud(z)] ;
	%avoid clicking 
    y1=[y1;zeros(441,2)];
    sounds_key{i}=y1;
end    
 
    
