function init_probe_tone_exp(opt)
% generates matrix of sequences for probe_tone_exp, divides it in block of
% opt.blocksize sequences and saves it in a cell array. by using the
% matrices in the cells of the cell array for the blocks of the probe tone experiments it can be made sure,
% that all combinations of keys and probetones are played. 

global VP_CODE



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

%Achtung! VP_CODE in Namen
fn=[VP_CODE '_sequences_matrices'];
save (fn, 'sequence_matrices');