function sequence= createSequence(N, nrSpeaker, varargin),
% if opt.repeatable is set to true, the returned sequence insures that the
% last block can be followed by the first block.
% opt.repetitionBreak determines how many other stimuli need to occur
% between two presentations of the same stimulus
% if opt.allowNeighbours is set to true, neighbouring speakers can follow
% eachother if N<5, this is enforced
% opt.possibleRounds limits the number of permutations of the speakers that
% are considered as possible rounds to save time

opt= propertylist2struct(varargin{:});

opt= set_defaults(opt, ...
    'repeatable', 0,...
    'repetitionBreak',2,...
    'allowNeighbours',0,...
    'possibleRounds',1000);

% setup the allowed options array
speaker_options = [];
for i=1:nrSpeaker,
    allowed = [[i+1:nrSpeaker],[1:i-1] ];
    % for more than 4 speakers, neighbouring speakers are not allowed
    if ~opt.allowNeighbours & nrSpeaker > 4,
        allowed = allowed(2:end-1);
    end
    speaker_options = [speaker_options;allowed];
end

% generate all possible rounds
all_rounds = perms([1:nrSpeaker]);

% limit the number of possibilities to save time
if size(all_rounds,1)>opt.possibleRounds,
    all_rounds = all_rounds(1+randint(opt.possibleRounds,1,size(all_rounds,1)),:);
end

% delete the ones that violate the conditions

delete_rounds = [];

for i=size(all_rounds,1):-1:1

    for j = 2:nrSpeaker
        % check if speaker is in the allowed followers of the previous one
        if isempty(find(all_rounds(i,j) == speaker_options(all_rounds(i,j-1),:))),
            delete_rounds =[delete_rounds,i];
            break
        end

    end

end

all_rounds(delete_rounds,:)=[];

n_rounds = size(all_rounds,1);
next_rounds = zeros(n_rounds,n_rounds);


% generate allowed connectivity array
for i=1:n_rounds,
    for j=1:n_rounds,

        next_rounds(i,j)=1;
        round1 = all_rounds(i,:);
        round2 = all_rounds(j,:);
        for k =1:nrSpeaker,
            pos1 = find(round1 == k);
            pos2 = find(round2 ==k);
            distance =nrSpeaker-pos1 +pos2-1;
            % if the distance between the same stimuli is less than
            % required, the connection is disabled
            if distance<opt.repetitionBreak,
                next_rounds(i,j)=0;
                break
            end
        end


    end
end


% randomly choose first round
round = randint(1,1,n_rounds)+1;
index_sequence=zeros(N,1);
index_sequence(1)=round;

% then randomly choose all the others from the array of allowed
% next rounds
for i =2:N,
    options = find(next_rounds(round,:));
    round = options(1+randint(1,1,length(options)));
    index_sequence(i)=round;

end

if opt.repeatable,
    % check if last block can be followed by first one
    if next_rounds(index_sequence(end),index_sequence(1)) == 0,
        % check if a different block fits between the second to last
        % and first rounds
        first_round = index_sequence(1);
        st_last_round = index_sequence(end-1);
        % get the list of possible last blocks
        possible = find(next_rounds(first_round,:));
        for i=1:length(possible),
            pos_round = possible(i);
            if next_rounds(pos_round,first_round),
                index_sequence(N)=pos_round;
                break
            end
        end
    end
    % it might not always work
    if  next_rounds(index_sequence(end),index_sequence(1)) == 0,
        warning('sequence is not repeatable')
    end
end




% generate the actual sequence from the index list

sequence = reshape(all_rounds(index_sequence,:)',N*nrSpeaker,1);

end

function x = randint(dimx, dimy, high)
    low = 0;
    x = floor(low + (high-low) * rand(dimx, dimy));
end
