function sequence = createSequence(N, nrSpeaker, varargin),
% if opt.repeatable is set to true, the returned sequence insures that the
% last block can be followed by the first block. 
% N can maximally be < 500. This function is implemented recursively and
% unless you change the limit imposed by Matlab, it will not allow for
% recursive depths larger than 500.

    opt= propertylist2struct(varargin{:});
    
    opt= set_defaults(opt, ...
                        'repeatable', 1);

    global BCI_DIR;
    
    %% for 5 speakers, use this hack because with strict rules the sequence
    %% will get into a repeating of 2 sequences
    if nrSpeaker == 5,
        posSer = [1, 3, 5, 2, 4; ...
                  1, 4, 2, 5, 3; ...
                  2, 4, 1, 3, 5; ...
                  2, 4, 1, 5, 3; ...
                  2, 5, 3, 1, 4; ...
                  3, 1, 4, 2, 5; ...
                  3, 5, 2, 4, 1; ...
                  3, 5, 1, 4, 2; ...
                  3, 1, 5, 2, 4; ...
                  4, 2, 5, 3, 1; ...
                  4, 2, 5, 1, 3; ...
                  4, 1, 3, 5, 2; ...
                  5, 3, 1, 4, 2; ...
                  5, 2, 4, 1, 3];

        posConn = [  6,  7,  8,  9, 13, 14; ...
                     3,  4,  5, 10, 11, 12; ...
                     1,  2,  4, 10, 11, 12; ...
                     2,  3,  5, 10, 11, 12; ...
                     6,  7,  8,  9, 13, 14; ...
                     1,  2,  9, 10, 11, 12; ...
                     3,  4,  5,  8, 13, 14; ...
                     1,  2,  6,  7,  9, 13; ...
                     1,  6,  7,  8, 13, 14; ...
                     3,  4,  5, 11, 13, 14; ...
                     3,  4,  5, 10, 12, 14; ...
                     1,  2,  6,  7,  8,  9; ...
                     1,  2,  6,  7,  8,  9; ...
                     3,  4,  5, 10, 11, 12];
    else
        load([BCI_DIR 'acquisition/stimulation/spatial_auditory_p300/sequenceData/' int2str(nrSpeaker) '.mat']);
    end
       
sequence = recursive_path([], N, nrSpeaker, posSer, posConn, opt);   
end

function newSeq = recursive_path(seq, N, nrSpeaker, posSer, posConn, opt),
    % depth reached
    if length(seq) == N,
        if opt.repeatable && ~is_repeatable(seq, posConn, opt),
            newSeq = [];
        else
            newSeq = reshape(posSer(seq, :)', 1, []);
        end
        return;
    end
    
    % initialize
    if isempty(seq),
        possible = 1:size(posConn, 1);
    else
        possible = posConn(seq(end), :);
        possible(isnan(possible)) = [];
    end
    possible = possible(randperm(length(possible)));
    newSeq = [];
    counter = 1;
    
    while isempty(newSeq) && counter <= length(possible),
        newSeq = recursive_path([seq possible(counter)], N, nrSpeaker, posSer, posConn, opt);
        counter = counter+1;
    end
end

function boolRep = is_repeatable(seq, posConn, opt),
    if ismember(seq(1), posConn(seq(end), :)),
        boolRep = true;
    else
        boolRep = false;
    end
end

    