function zoom_gui_exclusive
    % Create main figure
    fig = uifigure('Name', 'Zoom & Pan Exclusive GUI', 'Position', [100 100 700 450]);

    % Create axes
    ax = uiaxes(fig, 'Position', [75 100 550 300]);
    title(ax, 'Zoomable Plot');
    xlabel(ax, 'X');
    ylabel(ax, 'Y');

    % Initial plot
    x = linspace(0, 4*pi, 1000);
    y = sin(x);
    plot(ax, x, y);

    % Store original limits
    originalXLim = ax.XLim;
    originalYLim = ax.YLim;

    % Dropdown for plot type
    dd = uidropdown(fig, ...
        'Items', {'Sine', 'Cosine', 'Exponential Decay'}, ...
        'Position', [50 40 150 30], ...
        'ValueChangedFcn', @(dd, event) updatePlot(dd, ax));

    % Zoom button
    zoomBtn = uibutton(fig, ...
        'Position', [220 40 80 30], ...
        'Text', 'Zoom On', ...
        'ButtonPushedFcn', @(btn, event) toggleZoom(ax, btn));
    zoomBtn.UserData.isZoomOn = false;

    % Pan button
    panBtn = uibutton(fig, ...
        'Position', [320 40 80 30], ...
        'Text', 'Pan On', ...
        'ButtonPushedFcn', @(btn, event) togglePan(ax, btn));
    panBtn.UserData.isPanOn = false;

    % Store handles for access inside functions
    zoomBtn.UserData.panBtn = panBtn;
    panBtn.UserData.zoomBtn = zoomBtn;

    % Reset View button
    resetBtn = uibutton(fig, ...
        'Position', [420 40 100 30], ...
        'Text', 'Reset View', ...
        'ButtonPushedFcn', @(btn, event) resetView(ax, originalXLim, originalYLim));
end

function updatePlot(dd, ax)
    x = linspace(0, 4*pi, 1000);
    switch dd.Value
        case 'Sine'
            y = sin(x);
        case 'Cosine'
            y = cos(x);
        case 'Exponential Decay'
            y = exp(-0.2*x) .* sin(x);
    end
    plot(ax, x, y);
end

function toggleZoom(ax, btn)
    panBtn = btn.UserData.panBtn;
    if ~btn.UserData.isZoomOn
        zoom(ax, 'on');
        btn.Text = 'Zoom Off';
        btn.UserData.isZoomOn = true;

        % Turn off pan if it's on
        if panBtn.UserData.isPanOn
            pan(ax, 'off');
            panBtn.Text = 'Pan On';
            panBtn.UserData.isPanOn = false;
        end
    else
        zoom(ax, 'off');
        btn.Text = 'Zoom On';
        btn.UserData.isZoomOn = false;
    end
end

function togglePan(ax, btn)
    zoomBtn = btn.UserData.zoomBtn;
    if ~btn.UserData.isPanOn
        pan(ax, 'on');
        btn.Text = 'Pan Off';
        btn.UserData.isPanOn = true;

        % Turn off zoom if it's on
        if zoomBtn.UserData.isZoomOn
            zoom(ax, 'off');
            zoomBtn.Text = 'Zoom On';
            zoomBtn.UserData.isZoomOn = false;
        end
    else
        pan(ax, 'off');
        btn.Text = 'Pan On';
        btn.UserData.isPanOn = false;
    end
end

function resetView(ax, xlimDefault, ylimDefault)
    ax.XLim = xlimDefault;
    ax.YLim = ylimDefault;
end
