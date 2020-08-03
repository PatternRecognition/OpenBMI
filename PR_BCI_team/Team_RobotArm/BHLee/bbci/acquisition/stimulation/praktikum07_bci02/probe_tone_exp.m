function [opt mrp]=probe_tone_exp(varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Christina Müller, Irene Sturm 11/2007
% [opt mrp] = probe_tone_exp generates and plays all combinations of major and minor keys 
% with each tone of the chromatic scale. The tones are generates as Shepard's tones with the center
% of the amplitude envelope over the tonic of the key.
% 
% out:
% mrp               matrix with combination of key and probe tone in columns. can be
%                   used  to repeat experiment with the same order.
% opt               options structure
%
%[opt mrp]=probe_tone_exp('PARAM1',val1, 'PARAM2',val2,...)
%   specifies one or more of the following name/value pairs:
%
%     'mode'        mode in which the key is established, e.g. 'triad'
%                   (default) generates tonic, third and fifth, 'triad_plus_chord' generates a broken triad
%                   followed by a triad as chord, 'scale5' an ascending
%                    scale from tonic to fifth
%
%     'fs'   		samplerate
%                  (default 22100 Hz)
%
%     'tones'       Specifies how many partial tones are used for the
%                   Shepard's tones (default: 7)
%
%     'pause'		vector with three entries.the first entry specifies the
%                   duration of the gap between the key establishing sequence in s, the second
%                   specifies the pause between two sequences, the third the duration of the break between blocks of sequences (default [2 3 90]).
%
%	  'd'		    vector size(1,length(key establishing sequence +1) with value duration of each tone in seconds
%                   or scalar, if each tone has same duration (default 0.6 s for each tone)
%     'order'       'rand' (default) makes a random permutation of all
%                   sequences, 'double' plays each sequence of the random permutation
%                   twice, 'block' plays all probe tone combinations of each key blockwise.
%                   'block_plus' plays blockwise with three
%                   additional sequences at the beginning of a block. The
%                   value of 'order' can also be a [3 nSequences] matrix as
%                   produced by probe_tone_exp (mrp).
%  
%     'howmany'     how many sequences are played (for testing)
%
%Triggers:
% 1-24      : key e.g. 1 for C major, 2 for c minor, 3 for C# Major, ...
% 101-112   : probe tone stimulus: interval between tonic and probe tone in number of halftone steps
% 101-107   : responses
% examples: 
%
%
%
%[opt mrp]=probe_tone_exp('mode','triad_plus_chord','d',[.2 .2 .2 .6 .6],'tones',7,'pause',[2 3],'order','rand','howmany',2);
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         

%global STIMTRIGGER

%defaults
opt=propertylist2struct(varargin{:});
opt=set_defaults(opt,'mode','triad','pause',[2 3],'tones',7,'fs',22100,'order','rand','d',.6,'blocksize',72,'howmany','all','bv_host','localhost','require_response',0,'max_phi',0,'sigma',1,'fade',0.08);
mode=strmatch(opt.mode,{'triad','triad_plus_chord', 'scale7','scale5'},'exact');

p=opt.pause;


%halftone structures of all modes in major and minor, e.g. 0 3 7 for minor
%triad

sequences={{[0 4 7 ],[0 3 7]},{[0 4 7 0;0 4 7 4;0 4 7 7],[0 3 7 0;0 3 7 3;0 3 7 7]},{[0 2 4 5 7 9 11],[0 2 3 5 7 8 11]},{[0 2 4 5 7],[0 2 3 5 7]}};


%generate all combinations of keys and probetones
pos=1;m=[];
for i=0:11
    for j=1:2
        for k=0:11
            m(:,pos)=[i;j;k];
            pos=pos+1;
        end
    end
end



%arrange in specified random order or predefined order

    if (ischar(opt.order))
        switch(opt.order)
    
            case 'rand'
            [mrp]=get_random_order(m); 
            case 'block'
            [mrp]=get_block_order(m);
            case 'double'
            [mrp]=get_rand_double_order(m); 
            case 'block_plus'
            [mrp]=get_block_order_plus(m);
        end
    else
        mrp=opt.order;
    end

%get matrix to determine the marker ids for keys
matrix_id=construct_marker_id();

%durations
if (size(opt.d,2)==1)
    
    d_seq=opt.d*ones(1,size(sequences{mode}{1},2)+1);

    d_probe_tone=opt.d;
else
    if (size(opt.d,2)~=size(sequences{mode}{1},2)+1)
        error('length of opt.d does not match length of key establishing sequence plus probe tone')
    else 
        
        d_seq=opt.d(1:end-1);
        d_probe_tone=opt.d(end);
    end     
end

if isempty(~strmatch(opt.howmany,'all'))
    h=opt.howmany;
else
    h=size(mrp,2);
end    


if(h~=0)

%if isempty(state),
%  disp('init');
%  state= acquire_bv(1000, opt.bv_host);
%end

for i=1:h
    
    %generate key establishing sequence 
    
    fprintf('playing chord at freq %d: ', round(get_freq(mrp(1,i)))); 
    fprintf('%d ', sequences{mode}{mrp(2,i)}(1,1:3)); fprintf('\n');
    key=shep_chord(get_freq(mrp(1,i)),sequences{mode}{mrp(2,i)},'durations',d_seq,'tones',opt.tones,'fs',opt.fs);
    ppTrigger(get_id(matrix_id,mrp(1:2,i)));
    sound(key,opt.fs);
    
    pause(p(1));
    %generate and sound probe tone
    ppTrigger(100+mrp(3,i));
    sound(shep_chord(get_freq(mrp(1,i)),[mrp(3,i)],'durations',d_probe_tone,'tones',opt.tones,'fs',opt.fs),opt.fs);
    
    
   % get response markers
   
   if opt.require_response,
     STIMTRIGGER= [];
     while ~ismember(STIMTRIGGER,[101:107]) & 1000*etime(clock,t0)<p(2),
       pause(0.001);
     end
   else 
     pause(p(2))
   end
    
    
    end    
end