function graph = build_graph(lut_load)

    % assumes state(1) as 'root'. everything that is out of reach from
    % state(1) will not be included in the graph

    % store terminal nodes in graph.nodes
    % store edges in graph.edges (sparse)
    % terminal nodes have value > graph.termOffset
    
    stNode = lut_load(1);
    toParse = {1, [1:length(stNode.direction)]};
    
    graph.termOffset = length(lut_load)+1;
    graph.nodes = cell(0,2);
    
    sparFrom = zeros(1,100);
    sparTo = zeros(1,100);
    sparVal = zeros(1,100);
    
    escFlag = 0;
    libIdx = graph.termOffset;
    nodeIdx = 1;
    
    visitedNodes = [1];
    
    while ~isempty(toParse) && ~escFlag,
        % do single parse
        [do_state do_dir] = deal(toParse{1,1}, toParse{1,2}(1));
        
        %delete from queue
        toParse{1,2}(1) = [];
        if isempty(toParse{1,2}),
            toParse(1,:) = [];
        end
        
        % do the parse
        to_node = lut_load(do_state).direction(do_dir);
        switch to_node.type
            case 'navi'
                sparFrom(1,nodeIdx) = do_state;
                sparTo(1,nodeIdx) = to_node.nState;
                sparVal(1,nodeIdx) = do_dir;
                if ~ismember(to_node.nState, visitedNodes),
                    toParse{size(toParse, 1)+1, 1} = to_node.nState;
                    toParse{size(toParse, 1), 2} = [1:6];
                    visitedNodes = [visitedNodes to_node.nState];
                end
            case {'select', 'action'}
                graph.nodes(size(graph.nodes, 1)+1, :) = {libIdx, to_node.alt};
                sparFrom(1,nodeIdx) = do_state;
                sparTo(1,nodeIdx) = libIdx;
                sparVal(1,nodeIdx) = do_dir;              
                libIdx = libIdx + 1;
            otherwise
                error('Unknown node: %s', to_node.type);
        end     
        nodeIdx = nodeIdx + 1;
    end  
    sparFrom(sparFrom==0) = [];
    sparTo(sparTo==0) = [];
    sparVal(sparVal==0) = [];

    dimension = max([sparFrom, sparTo]);
    
    graph.edges = sparse(sparFrom, sparTo, sparVal, dimension, dimension);
end