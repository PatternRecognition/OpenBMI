function [handles, H, cn]= fb_handleStruct2Vector(HH, cc)

if nargin==1,
  cc= 0;
end

if ~isstruct(HH),
  handles= HH;
  cn= cc + length(HH);
  H= reshape([cc+1:cn], size(HH));
  return;
end

H= [];
handles= [];
fn= fieldnames(HH);
for jj= 1:length(HH),
  for ii= 1:length(fn),
    fld= getfield(HH, {jj}, fn{ii});
    if ~isstruct(fld),
      cn= cc + length(fld);
      handles= [handles, fld(:)'];
      H= setfield(H, {jj}, fn{ii}, reshape([cc+1:cn], size(fld)));
    else
      [recha,recH,cn]= fb_handleStruct2Vector(fld, cc);
      handles= [handles, recha];
      H= setfield(H, {jj}, fn{ii}, recH);
    end
  cc= cn;
  end
end
