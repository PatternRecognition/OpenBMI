% File Name: checkValidity.m
% Author: Heung-Il Suk
% Cite: H.-I. Suk and S.-W. Lee, "A Novel Bayesian Framework for Discriminative 
% Feature Extraction in Brain-Computer Interfaces," IEEE Trans. on PAMI,
% 2012. (Accepted)

function sample = opt_checkValidity( sample )
% We enforce the samples to be within min_freq and max_freq

min_freq = 0.05;
max_freq = 40;

if sample(1) <= min_freq
    sample(1) = min_freq;
end

if sample(2) <= min_freq
    sample(2) = min_freq;
end

if sample(1) > max_freq
    sample(1) = max_freq;
end

if sample(2) > max_freq
    sample(2) = max_freq;
end

if sample(1) > sample(2)
    temp = sample(2);
    sample(2) = sample(1);
    sample(1) = temp;
end

if (sample(1) - sample(2)) < 1
    sample(1) = sample(1);
    sample(2) = sample(2) + 0.5;
end


