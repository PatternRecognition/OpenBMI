opt=struct;

%options needed for probe_tone_exp;
opt.mode='triad_plus_chord';
opt.d=[.2 .2 .2 .6 .4];
opt.tones=9;
opt.pause=[2 2];
opt.order='rand';
opt.blocksize=72;
%opt.howmany=3; %for testing
opt.response_markers= {'R 101', 'R 102', 'R 103', 'R 104', 'R 105', 'R 106', 'R 107'};
opt.bv_host= 'localhost';
opt.require_response= 0;   %% this is for testing
opt.sequences_until_break=36;
opt.block=1;
opt.sigma=1;
opt.max_phi=0;
opt.fade=0.08;

%for logging response triggers
opt.keylist=['1234567'];
opt.res_offset=0;

%opt.position= [5 200 640 480];  %% for testing
opt.position= [-1919 0 1920 1181];
opt.cross= 1;
opt.countdown = 5;
opt.handle_background= stimutil_initFigure(opt);
opt.break_duration=15;

% generates matrix of sequences for probe_tone_exp, divides it in block of
% opt.blocksize sequences and saves it in a cell array. by using the
% matrices in the cells of the cell array for the blocks of the probe tone experiments it can be made sure,
% that all combinations of keys and probetones are played. 
[temp_opt mrp]=probe_tone_exp(opt,'howmany',0);

if (mod(288,opt.blocksize)~=0)
    fprintf('opt.blocksize is not a divisor of 288. Last block has less than opt.blocksize sequences.');
end

for k=1:(round(288/opt.blocksize))
    if (k==(round(288/opt.blocksize)))
         sequence_matrices{k}=mrp(:,(k-1)*opt.blocksize+1:end);
    else     
    sequence_matrices{k}=mrp(:,(k-1)*opt.blocksize+1:k*opt.blocksize);
    end
end


fn=[VP_CODE '_sequences_matrices'];
save (fn, 'sequence_matrices');

bn= 0;
fprintf('for testing:\n  bn= 0; stim_probe_tone( opt, ''test'',1);\n');
fprintf('bn= mod(bn,4)+1, stim_probe_tone(opt, ''block'',bn);\n');
