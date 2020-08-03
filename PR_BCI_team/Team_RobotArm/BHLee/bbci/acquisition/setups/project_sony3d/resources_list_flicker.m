freqs = [50 60];
bright = {'low', 'high', 'medium', 'medium with towel on sender', 'goggles on front'};
stimuli = {...
    'circle_size_0.1_grey_0.2'
    'circle_size_0.1_grey_0.6'
    'circle_size_0.1_grey_1.0'
    'circle_size_0.5_grey_0.2'
    'circle_size_0.5_grey_0.6'
    'circle_size_0.5_grey_1.0'
    'circle_size_0.8_grey_0.2'
    'circle_size_0.8_grey_0.6'
    'circle_size_0.8_grey_1.0'
    'jungle_bright'
    'jungle_dark'
    'jungle_normal'
    };

% generate markers
marker_black = 10;
markers = ( 1 : numel(freqs)*numel(bright)*numel(stimuli) ) + marker_black;
markers = reshape(markers, [numel(freqs) numel(bright) numel(stimuli)]);