function dat = proc_selectHemisphere(dat, IdString)
%dat= proc_selectChannels(dat, chans)
%
% IN   dat      - data structure of continuous or epoched data
%      IdString - single region or cell array with list of regions to be selected,
%                 e.g. 'left','right',occipital','frontal'
%                 if multiple regions are given the intersection
%                 will be selected, e.g. {'left','right'} will
%                 yield in the central line '*z'
%
% OUT  dat   - updated data structure
%
% SEE also proc_selectChannels, chanind

% stl, Berlin, Sep.2004, ida.first.fhg.de


if isempty(IdString) | ~(ischar(IdString)|iscell(IdString)),
  error('invalid or missing argument IdString in dat = proc_selectHemisphere(dat, IdString)');
end ;

if ischar(IdString),
  IdString = {IdString};
end 
nStrings = length(IdString) ;

for idx = 1: nStrings,
  switch lower(IdString{idx}),
   case 'left'
    dat = proc_selectChannels(dat, {'*z','*1','*3','*5','*7','*9'});
   case 'right'
    dat = proc_selectChannels(dat, {'*z','*2','*4','*6','*8','*10'});
   case 'occipital'
    dat = proc_selectChannels(dat, {'C#','T*','CC*','CP*','FC*','P*','O*','I*'});
   case 'frontal'
    dat = proc_selectChannels(dat, {'C#','CF*','F*','AF*'});
   otherwise
    warning(sprintf('Ignoring unknown or not yet implemented brain region: %s !!!',upper(IdString{idx})));
  end ;
end ;

