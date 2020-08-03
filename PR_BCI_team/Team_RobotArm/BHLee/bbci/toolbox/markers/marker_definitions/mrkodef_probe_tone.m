function mrk= mrkodef_probe_tone(mrk_orig)

%% define marker struct
pitch_class_list= {'C','C#','D','Eb','E','F','F#','G','G#','A','Bb','B'};
minor_list= lower(pitch_class_list);
key_list= strcat(cat(1, pitch_class_list, minor_list), ...
                 repmat({'-major';'-minor'},[1 12]));
key_list= key_list(:)';
contextDef= cat(1, cprintf('S%3d', 1:24)', key_list);
mrk_context= mrk_defineClasses(mrk_orig, contextDef);
imajor= find(mod(mrk_context.toe,2)==1);
iminor= find(mod(mrk_context.toe,2)==0);
mrk_context.keyno(imajor)= (mrk_context.toe(imajor)+1)/2;
mrk_context.keyno(iminor)= 12+mrk_context.toe(iminor)/2;
mrk_context= mrk_addIndexedField(mrk_context, 'keyno');

%This was for season1:
%probeDef= cat(1, cprintf('S%3d', 100:111')', pitch_class_list);
probeDef= cat(1, cprintf('S%3d', 101:112')', pitch_class_list);
mrk_probe= mrk_defineClasses(mrk_orig, probeDef);
mrk_probe.toe= mrk_probe.toe-100;
if length(mrk_context.pos)~=length(mrk_probe.pos),
  error('trouble');
% To find the problem:
%  [mrk_tmp, ic, ip]= mrk_matchStimWithResp(mrk_context, mrk_probe);
end
%This was for season1:
%responseDef= cat(1, cprintf('R%3d', 101:107')', cprintf('%d', 1:7)');
responseDef= cat(1, cprintf('R%3d', 2.^[0:6]')', cprintf('%d', 1:7)');
mrk_resp= mrk_defineClasses(mrk_orig, responseDef, 'removevoidclasses', 0);
mrk_resp.toe= abs(mrk_resp.toe)-100;

[mrk, istim, iresp]= ...
    mrk_matchStimWithResp(mrk_probe, mrk_resp, ...
                          'missingresponse_policy', 'accept',...
                          'multiresponse_policy', 'last', ...
                          'removevoidclasses', 0);
mrk.context= mrk_context;
mrk.probe= mrk_probe;
