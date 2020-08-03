function correctSel = findCorrectStep(graph, currNode, tarChar),
    
    if length(tarChar) == 1,
        tarChar = upper(tarChar);
    end
    tarIdx = find(strcmp(graph.nodes(:,2), tarChar));
    if isempty(tarIdx),
        error(' ''%s'' not in dictionary', tarChar);
    end
    tarNode = graph.nodes{tarIdx,1};

    if tarNode == currNode,
        correctSel = [];
        return;
    end

    termNode = [currNode];
    path = {[]};
    visited = [];

    while ~isempty(termNode),
        % find children of current node
        chNode = find(graph.edges(termNode(1),:));
        chPath = full(graph.edges(termNode(1), chNode));

        for i = 1:length(chNode),
            if ~ismember(chNode(i), visited),
                termNode(length(termNode)+1) = chNode(i);
                visited = [visited chNode(i)];
                path{length(path)+1} = [path{1} chPath(i)];
            end
        end

        termNode(1) = []; path(1) = [];
        
        if ismember(tarNode, termNode),
            corId = find(termNode == tarNode);
            correctSel = path{corId};
            break;
        end
    end

    if isempty(correctSel),
        warning(' ''%s'' does not appear in the graph', tarChar);
    end
        
end