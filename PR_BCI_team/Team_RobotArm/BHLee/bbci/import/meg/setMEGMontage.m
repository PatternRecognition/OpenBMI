function [mnt] = setMEGMontage(clab,varargin) ;
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
  [varargin{1}, varargin{2}]= calc_pos_MEG306;
end

if ischar(varargin{1}),
  posFile= ['calc_pos_' varargin{1}];
  [posSystem3D, posSystem2D] = feval(posFile);
  displayMontage= {varargin{2:end}};
elseif isstruct(varargin{1}),
  [posSystem3D]= varargin{1};
  [posSystem2D]= varargin{2};
  displayMontage= {varargin{3:end}};
end

elab= posSystem3D.clab;

nanIdx =  [] ;
%nanIdx = find(sqrt(sum([posSystem2D.x posSystem2D.y].^2,2))>34) ;
posSystem2D.x(nanIdx) = NaN ;
posSystem2D.y(nanIdx) = NaN ;

nChans= length(clab);
mnt.x= NaN*ones(nChans, 1);
mnt.y= NaN*ones(nChans, 1);
mnt.pos_3d= NaN*ones(3, nChans);
for ei= 1:nChans,
  ii= chanind(elab, clab{ei});
  if ~isempty(ii),
    mnt.x(ei)= posSystem2D.x(ii);
    mnt.y(ei)= posSystem2D.y(ii);
    mnt.pos_3d(:,ei)= [posSystem3D.x(ii) posSystem3D.y(ii) posSystem3D.z(ii)];
  end
end

radius_x = max(abs(mnt.x)) ;
radius_y = max(abs(mnt.y)) ;
mnt.x= .87*mnt.x/radius_x ;
mnt.y= .87*mnt.y/radius_y ;
mnt.clab= clab ;

mnt= setDisplayMontage(mnt, displayMontage{:});







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% subfunctions %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [posSystem3D, posSystem2D] = calc_pos_MEG122 ; 
  global MEG_RAW_DIR  MEG_CFG_DIR ; 
  cname = 'NM122coildef.mat' ;
  
  [cLab,cId,cNu]=channames([MEG_CFG_DIR 'Emptyroom_avg.fif']) ;
  cLab(cId ~= 1) = [] ;
  cLab = strrmspace(cLab) ;

  load(cname) ;        
  coilIdx = reshape([1:61;1:61],122,1) ;
  posSystem3D.x = VM(coilIdx,1)' ;
  posSystem3D.y = VM(coilIdx,2)' ;
  posSystem3D.z = VM(coilIdx,3)' ;
  pos.System3D.clab = cLab ;
  
  
function [posSystem3D, posSystem2D] = calc_pos_MEG306 ; 
  global MEG_RAW_DIR  MEG_CFG_DIR ; 
  cname = 'NM306coildef.mat' ;
  load(cname) ;         

  [cLab,cId,cNu]=channames([MEG_CFG_DIR 'Emptyroom_avg.fif']) ;
  cLab(cId ~= 1) = [] ;
  cLab = strrmspace(cLab) ;

  if length(cLab) ~= 306, 
    error('improper number of channels read from file' ) ;
  end ;

  coilIdx = reshape([1:102;1:102;1:102],306,1) ;
  posSystem3D.x     = VM(coilIdx,1)' ;
  posSystem3D.y     = VM(coilIdx,2)' ;
  posSystem3D.z     = VM(coilIdx,3)' ;
  posSystem3D.clab = cLab ;

  cname = 'c102.dat' ;
  load(cname) ;
  
  for ch = 1: size(c102,1),
    if c102(ch,6) < 1000 ,
     cLab2D{ch} = ['MEG0' num2str(c102(ch,6))] ;
    else 
      cLab2D{ch} = ['MEG' num2str(c102(ch,6))] ;
    end ;
  end ;
  
  yChans = chanind(cLab2D,{'MEG0811','MEG2111'}) ;
  xChans = chanind(cLab2D,{'MEG1331','MEG0241'}) ;

  adjust = mean([c102(xChans,2) c102(yChans,3)]) 
  
  posSystem2D.x = c102(coilIdx,2)-adjust(1) ;
  posSystem2D.y = c102(coilIdx,3)-adjust(2) ;
  posSystem2D.clab = cLab ;




