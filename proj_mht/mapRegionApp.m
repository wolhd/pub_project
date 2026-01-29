function MapClearApp
    % 1. Create Figure
    fig = uifigure('Name', 'Interactive Map Capture', 'Position', [100 100 800 600]);
    
    % 2. Setup 2x2 Grid: Row 1 is stretchy, Row 2 fits buttons
    gl = uigridlayout(fig, [2 2]);
    gl.RowHeight = {'1x', 'fit'};
    gl.ColumnWidth = {'1x', 'fit'};
    gl.Padding = [10 10 10 10];

    % 3. Map Axes (Spans everything)
    gax = geoaxes(gl);
    gax.Layout.Row = [1 2];    
    gax.Layout.Column = [1 2]; 
    title(gax, 'Capture Controls in Bottom-Right');

    % 4. Sub-container for Buttons (Row 2, Col 2)
    % This keeps them grouped together in the corner
    btnGroup = uigridlayout(gl, [1 2]);
    btnGroup.Layout.Row = 2;
    btnGroup.Layout.Column = 2;
    btnGroup.RowHeight = {'fit'};
    btnGroup.ColumnWidth = {'fit', 'fit'};
    btnGroup.Padding = [5 5 5 5];

    hROI = []; 

    % Capture Button
    uibutton(btnGroup, 'Text', 'Capture', ...
        'ButtonPushedFcn', @(btn, event) startCapture(gax));
    
    % Clear Button
    uibutton(btnGroup, 'Text', 'Clear', ...
        'ButtonPushedFcn', @(btn, event) clearCapture());

    % --- Callbacks ---
    function startCapture(ax)
        clearCapture(); 
        hROI = drawrectangle(ax, 'Color', 'r', 'Label', 'Capture Area');
        addlistener(hROI, 'ROIMoved', @(src, evt) updateBounds(src.Position));
        updateBounds(hROI.Position);
    end

    function clearCapture()
        if ~isempty(hROI) && isvalid(hROI)
            delete(hROI);
            hROI = [];
        end
    end

    function updateBounds(pos)
        % [LonMin, LatMin, LonWidth, LatHeight]
        lonLim = [pos(1), pos(1) + pos(3)];
        latLim = [pos(2), pos(2) + pos(4)];
        fprintf('Bounds -> Lat: [%.2f, %.2f], Lon: [%.2f, %.2f]\n', ...
            latLim(1), latLim(2), lonLim(1), lonLim(2));
    end
end
