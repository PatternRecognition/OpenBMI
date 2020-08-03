function [sequence,rc_sequence] = createRowColumnSequences(N,numRows,numCols,stimulusSeparation)
% creates a pseudo-random sequence of stimuli for a row-column style matrix speller (e.g. photobrowser)
% INPUT:
%      N (int):                     number of rounds/frames (during one round/frame, every column and every row is highlighted once)
%      stimulusSeparation (int):    after this number of other stimuli a row or column can be repeated.
%      numCols, numRows (both int): size of stimulus matrix.
% 
% Output: 
%   sequence: vector of ints that index the single positions in the
% matrix. All entries belonging to a single row or column are concatenated.
% The final application has to know, how many entries a row or column has,
% and how many items are containted in one round.
%
%   rc_sequence: vector of int, that index the rows and colums only
% instead of indexing all single elements.

nrStimuli = numCols+numRows;

% initialise sequence array
sequence = zeros(N*nrStimuli,1);
lastround = randperm(nrStimuli);
sequence(1:nrStimuli,1)=lastround;

for n=1:N-1,
    
    
    % generate each round until separation conditions are fullfilled
    correct = 0;
    while ~correct,
        round = randperm(nrStimuli);
        correct = 1;
        
        for i=1:stimulusSeparation,
           current = round(i);
           endbit = lastround(end - stimulusSeparation+i:end);
           if sum(endbit==current)>0,
               correct = 0;
           end
        end
        
    end
    lastround = round;
    sequence(n*nrStimuli+1:(n+1)*nrStimuli,1)=round;
end

rc_sequence=sequence;

% Convert Sequence to describe row-/column position on a
% speller/photobrowser matrix (works for square and non-square matrices, but stops with an error for non-square matrices due to photobrowser implementation)

% Sanity Check
if (min(sequence)<1 || max(sequence)>(numRows+numCols)) || numRows~=numCols
    error('Error in translating row-column sequence into matrix positions. Check entries of sequence. Check if matrix is square.')
end

matrixLUT=1:(numRows*numCols);
matrixLUT=reshape(matrixLUT,numCols,numRows)'

sequence_m=[]; %blown-up sequence for matrix

for i=1:length(sequence)
    if sequence(i) <= numRows
        sequence_m= [sequence_m ,matrixLUT(sequence(i),:)];
    else
        sequence_m= [sequence_m, matrixLUT(:,sequence(i)-numRows)'];
    end
end

sequence=sequence_m';