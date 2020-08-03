function [mnt, cx , cy] = defineMontage(clab,varargin) ;
%mnt= setElectrodeMontage(clab, <<posSystem>, displayMontage>)
%mnt= setElectrodeMontage(clab, alpha, beta, <displayMontage>)
%mnt= setElectrodeMontage(clab, x, y, z, <displayMontage>)
%
% IN   clab  
%      posSystem      - name such that ['calc_pos_ ' posSystem] is an m-file
%                       or a struct with fields x, y, z, clab
%      displayMontage - name for grid view layout, see setDisplayMontage
%
% OUT  mnt       struct for electrode montage
%         .x     - x coordiante of electrode positions
%         .y     - y coordinate of electrode positions
%         .clab  - channel labels
%         .box_x - x coordinate of axes for grid view
%         .box_y - y coordinate of axes for grid view
%         .box_w - width of axes for grid view
%         .box_h - height of axes for grid view
%
% SEE  setDisplayMontage, calc_ext_10_10_pos

global MEG_RAW_DIR;

if nargin==1 | isempty(varargin{1}),
  varargin{1}= calc_pos_ext_10_10;
end

if ischar(varargin{1}),
  posFile= ['calc_pos_' varargin{1}];
  if exist(posFile, 'file'),
    posSystem= feval(posFile);
    displayMontage= {varargin{2:end}};
  else
    posSystem= calc_pos_ext_10_10;
    displayMontage= {varargin{:}};
  end
  x= posSystem.x;
  y= posSystem.y;
  z= posSystem.z;
  elab= posSystem.clab;
elseif isstruct(varargin{1}),
  posSystem= varargin{1};
  displayMontage= {varargin{2:end}};
  x= posSystem.x;
  y= posSystem.y;
  z= posSystem.z;
  elab= posSystem.clab;
elseif nargin==5 | (nargin==4 & ~ischar(varargin{3})),
  elab= clab;
  [x,y,z]= deal(varargin{1:3});
  displayMontage= {varargin{4:end}};
else
  elab= clab;
  [x,y,z]= abr2xyz(varargin{1:2});
  displayMontage= {varargin{3:end}};
end

maz= max(z(:));
miz= min(z(:));
%ur= [0 0 miz-0.8*(maz-miz)];
ur= [0 0 -1.5];

la= (maz-ur(3)) ./ (z(:)-ur(3));
Ur= ones(length(z(:)),1)*ur;
pos2d= Ur + (la*ones(1,3)) .* ([x(:) y(:) z(:)] - Ur);
pos2d= pos2d(:, 1:2);
pos2d(z<0,:)= NaN;

nChans= length(clab);
mnt.x= NaN*ones(nChans, 1);
mnt.y= NaN*ones(nChans, 1);
mnt.pos_3d= NaN*ones(3, nChans);
for ei= 1:nChans,
  ii= chanind(elab, clab{ei});
  if ~isempty(ii),
    mnt.x(ei)= pos2d(ii, 1);
    mnt.y(ei)= pos2d(ii, 2);
    mnt.pos_3d(:,ei)= [x(ii) y(ii) y(ii)];
  end
end
radius= 1.9;
mnt.x= mnt.x/radius;
mnt.y= mnt.y/radius;
mnt.clab= clab;

mnt= setDisplayMontage(mnt, displayMontage{:});







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% subfunctions %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [posSystem] = calc_pos_MEG122 ; 
  cname = 'NM122coildef.mat' ;
  
  coilIdx = reshape([1:61;1:61],122,1) ;
  load(cname) ;        
  posSystem.x = VM(coilIdx,1)' ;
  posSystem.y = VM(coilIdx,2)' ;
  posSystem.z = VM(coilIdx,3)' ;
  
  
function [posSystem] = calc_pos_MEG306 ; 
  cname = 'NM306coildef.mat' ;
  load(cname) ;         

  [cLab,cId,cNu]=channames([MEG_RAW_DIR 'demo/demo.fif']) ;
  cLab(cId ~= 1) = [] ;
  if length(cLab ~= 306), 
    error('improper number of channels read from file' ) ;
  end ;
  
  posSystem.x     = VM(coilIdx,1)' ;
  posSystem.y     = VM(coilIdx,2)' ;
  posSystem.z     = VM(coilIdx,3)' ;
  pos.System.clab = cLab ;



