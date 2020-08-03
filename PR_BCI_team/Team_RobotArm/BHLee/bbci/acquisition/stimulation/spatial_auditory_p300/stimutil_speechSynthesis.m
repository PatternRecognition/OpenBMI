function varargout = stimutil_speechSynthesis(textToSpell, varargin),

% Synopsis: this function takes strings as an input and uses the open
% source text-to-speech software to create audio fragments from them.
%
% use:
%    speech = stimutil_speechSynthesis(textToSpell, opt);
%    [speech, handle] = stimutil_speechSynthesis(textToSpell, opt);
%
% INPUT
%    textToSpell     Cell array with the different texts that need
%                    to be converted to speech
%    OPT
%     .voice           Name of the voice to be used [default:
%                      'hmm-bits2']
%     .procVar.spchHandle  Handle to the Java object containing the
%                      speech synthesis functions
%     .languageIndicator  Set language indicator for the Mary software
%                      [default: 'TEXT']. English = 'TEXT_EN', German
%                      = 'TEXT_DE'
%
% OUTPUT
%    speech          Cell array with the speech data array. Size =
%                    size(textToSpell)
%    handle          Handle to the Java object containing the speech
%                    synthesis functions
%
% Some notes to get this to work
% 1) Install MARY TTS from http://mary.dfki.de/download/mary-install-3.x.x.jar
% 2) Add the C:\Programme\MARY TTS\java\maryclient.jar to the
%    classpath.txt file (edit claspath.txt) (dir may of course be
%    different).
% 3) Restart matlab
% 4) Start the MARY server
% 5) Only then a call to this function *MIGHT* work
%
% Does some lowpass filtering to prevent the squicking sounds.
%
% Martijn Schreuder, 11/08/2009
    
    import java.io.*
    import de.dfki.lt.mary.client.*

    opt= propertylist2struct(varargin{:});
    opt= set_defaults(opt, ...
        'voice', 'hmm-bits2', ...
        'languageIndicator', 'TEXT_DE');
    
    global TMP_DIR;
    if ~exist(TMP_DIR, 'dir'),
        mkdir(TMP_DIR);
    end
    
    file = [TMP_DIR 'temp.wav'];

    if isfield(opt, 'procVar') &&isfield(opt.procVar, 'spchHandle') && ~isempty(opt.procVar.spchHandle),
        handle = opt.procVar.spchHandle;
    else
        handle = MaryClient('localhost', 59125);
    end   

    if ~iscell(textToSpell),
        textToSpell = {textToSpell};
    end
    wavOut = cell([1, length(textToSpell)]);
    
%     [filt.b filt.a] = butter(8, 3000/16000*2, 'low');
    for i = 1:length(textToSpell),
        output = FileOutputStream(file);
        try
            handle.process(textToSpell{i}, opt.languageIndicator, 'AUDIO', 'WAVE', opt.voice, '', '', output);
            wavOut{i} = wavread(file);
%             wavOut{i} = filter(filt.b, filt.a, wavOut{i});
            wavOut{i} = resample(wavOut{i}, 44100, 16000, [1]);
        catch
            warning(['The following sentence could not be parsed: ' textToSpell{i}]);
        end
        output.close();
    end
    
    varargout{1} = wavOut;
    if nargout == 2,
        varargout{2} = handle;
    end

end
