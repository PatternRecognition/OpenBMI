function [cli, cli_ind, fill_ins] = chanCliques(mnt, varargin)
% function [cli, cli_ind, fill_ins] = chanCliques(mnt, <opt>)
%
% get a cell array of Cliques with channels from mnt.clab; according to descriptor.
% 
% IN mnt        - struct describing the electrode montage. must contain
%                 fields .clab and .pos_3d.
%    opt        - options struct with possible fields:
%          descriptor - string describing the cliques setup, or cell array.
%                       possible string values: 'LRsmall','LRFsmall',~'large'.
%                       default: {}
%          max_dist   - defines the maximum distance of a channel to its neighbors.
%                       default 0 (changes nothing).
%          max_length - defines the maximum size of the neighborhood.
%                       default 9. note: this is done after max_dist.
%          bad_channels- kick out those channels.
%          triangulate- if true, murphy's triangulate-algorithm is used.
%          join_neighbors - if true, join the neighborhoods of clique centers.
%    
% 
% OUT cli        - cell array containing cliques of channels.
%     cli_ind    - cell array containing indices of these channels.
%     fill_ins   - matrix of additional entered edges.
%
% uses cliques2graph

% kraulem 08/04

if nargin>1
  opt = varargin{1};
else
  opt = struct('descriptor',{cell(0)});
end
opt = set_defaults(opt, 'descriptor', {cell(0)},...
                        'max_dist', 0,...
                        'max_length', 9,...
                        'bad_channels',{'E*'},...
                        'triangulate',0,...
			'join_neighbors',0);

if ischar(opt.descriptor)
  switch opt.descriptor
   case 'LRsmall'
    cli_centers = {{'C5' 'C3' 'C1'},...
                  {'C6' 'C4' 'C2'}};
   case 'LRFsmall'
    cli_centers = {{'C5' 'C3' 'C1'},...
                  {'C6' 'C4' 'C2'},...
                  {'CPz' 'Cz' 'FCz' 'POz'}};
   case 'LRlarge'
    cli_centers = {{'FC5' 'FC3' 'FC1' 'CFC5' 'CFC3' 'CFC1' 'C5' 'C3' ,...
                    'C1' 'CCP5' 'CCP3' 'CCP1' 'PCP5' 'PCP3' 'PCP1'},...
                  {'FC6' 'FC4' 'FC2' 'CFC6' 'CFC4' 'CFC2' 'C6' 'C4' ,...
                   'C2' 'CCP6' 'CCP4' 'CCP2' 'PCP6' 'PCP4' 'PCP2'}};
   case 'LRFlarge'
    cli_centers = {{'FC5' 'FC3' 'FC1' 'CFC5' 'CFC3' 'CFC1' 'C5' 'C3' ,...
                    'C1' 'CCP5' 'CCP3' 'CCP1' 'PCP5' 'PCP3' 'PCP1'},...
                  {'FC6' 'FC4' 'FC2' 'CFC6' 'CFC4' 'CFC2' 'C6' 'C4' ,...
                   'C2' 'CCP6' 'CCP4' 'CCP2' 'PCP6' 'PCP4' 'PCP2'},...
                  {'FCz' 'FC1' 'FC2' 'C1' 'C2' 'CP1' 'CP2' 'P1' 'P2'}};
   otherwise
    disp('unknown descriptor!');
  end
else
  cli_centers = opt.descriptor;
end
clab_ind = chanind(mnt.clab,'not',opt.bad_channels{:});
clab = {mnt.clab{clab_ind}};

% the neighborhood consists of the (physically) nearest neighbors.
neighbor_ch = {};
for i = clab_ind
  dist_arr = zeros(length(mnt.clab));
  for j = clab_ind
    dist_arr(j) = norm(mnt.pos_3d(:,i)-mnt.pos_3d(:,j));
  end
  [a,b] = sort(dist_arr(clab_ind));
  if opt.max_dist
    b = b(find(a<opt.max_dist));
  end
  if opt.max_length
    b = b(1:min(opt.max_length,end));
  end
  neighbor_ch{i} = {mnt.clab{clab_ind(b)}};
end

clab_ind = 1:length(clab);

neighbor_ind = {};
for i = clab_ind
  neighbor_ind{i} = chanind(clab, neighbor_ch{i}{:});
end

if opt.join_neighbors
  cli = {};
  cli_ind = {};
  % join some neighborhoods to cliques as required.
  for i = 1:length(cli_centers)
    cli_centers_ind = chanind(clab, cli_centers{i}{:});
    cli_ind{i} = unique([neighbor_ind{cli_centers_ind}]);
    cli{i} = clab(cli_ind{i});
  end
else
  cli = cli_centers;
end
  
% generate a graph out of the cliques:
adj = cliques2graph(cli,clab);

% now insert the neighborhood relations:
for i = 1:length(neighbor_ind)
  adj(neighbor_ind{i}(1),neighbor_ind{i}(2:end)) = 1;
  adj(neighbor_ind{i}(2:end),neighbor_ind{i}(1)) = 1;
end

% translate this graph to cliques:
if opt.triangulate
  order = best_first_elim_order(adj,2*ones(size(adj,1),size(adj,2)));
  [dum,cli_ind, fill_ins] = triangulate(logical(adj),order);
else
  cli_ind = graph2cliques(logical(adj));
  fill_ins = zeros(size(adj,1),size(adj,2));
end
cli = {};
for i =1:length(cli_ind)
  cli{i} = clab(cli_ind{i});
end

