function block_info = generate_experiment_blocks(nTrials,lb,mb,rb)
% Generiert die Abfolge von Experiment parametern : Stimuli, SOA und Cue-interval

%b = generate_experiment_blocks(6,3.5,4,5.5)

tones = [ 1,3 ,4,6 ,7,9 ];

SOAs = [150, 175, 200, 200, 225];

% M(i,b) enthaelt die SOA von Stimulus i in Block b
M = zeros(nTrials,5);
for k=1:nTrials
    M(k,:) = SOAs(randperm(5));
end

n_blocks = 5;
block_info = cell(n_blocks,3);
for b=1:n_blocks
    % zufaellige Reihenfolge von Stimuli in diesem Block
    stim_sequence = randperm(nTrials);
    block_info{b,1} = stim_sequence;
    % die SOA zu jeden Stimulus
    block_info{b,2} = M(stim_sequence, b)';
    % Cue-interval in diesem Block
    block_info{b,3} = lin_sample(nTrials,lb,mb,rb);
    
    %correct stim indexes for second part
    if nTrials == 6 
        block_info{b,1} = tones(block_info{b,1});
    end
    
end


