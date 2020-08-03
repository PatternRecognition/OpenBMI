function mrk= mrkodef_imag_fbarrow(mrko, varargin)

opt= propertylist2struct(varargin{:});
[opt, isdefault] = set_defaults(opt, ...
                  'classes_fb', {'target 1','target 2'}, ...
                  'stim_desc', [200 201], ...
                  'number_classifiers', 5, ...
                  'special_marker', 99, ...
                  'do_security_check', 1, ...
                  'multiresponse_policy', 'first', ...
                  'marker_offset', 0);
              
stimDef= {opt.stim_desc(1), opt.stim_desc(2);
          opt.classes_fb{:}};
respDef= getResponseDefinitions(opt.number_classifiers, opt.special_marker, opt.marker_offset);
miscDef= {70, 251, 254, 75, 76;
          'start countdown', 'begin block', 'end block', ...
          'start trial', 'end trial'};

opt= set_defaults(opt, 'stimDef', stimDef, ...
                       'respDef', respDef, ...
                       'miscDef', miscDef);

mrk_stim= mrk_defineClasses(mrko, opt.stimDef);
mrk_resp= mrk_defineClasses(mrko, opt.respDef);
mrk_misc= mrk_defineClasses(mrko, opt.miscDef);

[mrk, istim, iresp]= mrk_matchStimWithResp(mrk_stim, mrk_resp, 'removevoidclasses', 0, 'multiresponse_policy', opt.multiresponse_policy);
mrk_resp.pos = mrk_resp.pos(iresp);
mrk_resp.toe = mrk_resp.toe(iresp);
mrk_resp.y = mrk_resp.y(:,iresp);

mrk= rmfield(mrk, 'latency');
mrk = setHits(mrk, mrk_resp, opt.number_classifiers, opt.special_marker);
mrk.indexedByEpochs= {'resp_toe', 'ishit','isdraw', 'multihit'};
mrk.misc= mrk_misc;

end

function respDef = getResponseDefinitions(number_classifiers, special_marker, marker_offset),
    decMark = [0:2^number_classifiers-1];
    binMark = dec2bin(decMark);
    respDef = {};
    for i = 1:size(binMark, 1),
        respDef{1,i} = decMark(i)+marker_offset;
        respDef{2,i} = binMark(i,:);
    end
    respDef{1,1} = special_marker;
end

function mrk = setHits(mrk, mrk_resp, number_classifiers, special_marker),
    ishit = zeros(1,length(mrk.toe));
    isdraw = zeros(1,length(mrk.toe));
    multihit = zeros(number_classifiers, length(mrk.toe));
    dec_plane = number_classifiers/2;
    
    for i = 1:length(ishit),
        tmpMrk = mrk_resp.className{find(mrk_resp.y(:,i))};
        tmpMrk = str2num(tmpMrk(:));
        voting = sum(tmpMrk);
        switch mrk.toe(i),
            case 200
                ishit(i) = voting < dec_plane;
                isdraw(i) = voting == dec_plane;
                multihit(:,i) = ~tmpMrk;
            case 201
                ishit(i) = voting > dec_plane;
                isdraw(i) = voting == dec_plane;
                multihit(:,i) = tmpMrk;
        end
    end
    mrk.ishit = ishit;
    mrk.multihit = multihit;
    mrk.isdraw = isdraw;
end