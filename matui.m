function zoom_gui_scroll_toggle
    % Create the main UI figure
    fig = uifigure('Name', 'Zoom & Pan GUI with Scroll Toggle', ...
                   'Position', [100 100 700 450]);

    % Create axes
    ax = uiaxes(fig, 'Position', [75 100 550 300]);
    title(ax, 'Zoomable Plot');
    xlabel(ax, 'X');
    ylabel(ax, 'Y');

    % Initial plot
    x = linspace(0, 4*pi, 1000);
    y = sin(x);
    plot(ax, x, y);

    % Store default view
    originalXLim = ax.XLim;
    originalYLim = ax.YLim;

    % Dropdown for plot selection
    dd = uidropdown(fig, ...
        'Items', {'Sine', 'Cosine', 'Exponential Decay'}, ...
        'Position', [50 40 150 30], ...
        'ValueChangedFcn', @(dd, event) updatePlot(dd, ax));

    % Zoom button
    zoomBtn = uibutton(fig, 'Position', [220 40 80 30], ...
        'Text', 'Zoom On', ...
        'ButtonPushedFcn', @(btn, event) toggleZoom(ax, btn, fig));
    zoomBtn.UserData.isZoomOn = false;

    % Pan button
    panBtn = uibutton(fig, 'Position', [320 40 80 30], ...
        'Text', 'Pan On', ...
        'ButtonPushedFcn', @(btn, event) togglePan(ax, btn, fig));
    panBtn.UserData.isPanOn = false;

    % Link buttons
    zoomBtn.UserData.panBtn = panBtn;
    panBtn.UserData.zoomBtn = zoomBtn;

    % Reset button
    resetBtn = uibutton(fig, ...
        'Position', [420 40 100 30], ...
        'Text', 'Reset View', ...
        'ButtonPushedFcn', @(btn, event) resetView(ax, originalXLim, originalYLim));

    % Enable scroll zoom callback
    fig.WindowScrollWheelFcn = @(src, event) handleScrollZoom(ax, fig, event);

    % Set initial zoom state in UserData
    fig.UserData.isZoomOn = false;
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

function toggleZoom(ax, btn, fig)
    panBtn = btn.UserData.panBtn;
    if ~btn.UserData.isZoomOn
        zoom(ax, 'on');
        btn.Text = 'Zoom Off';
        btn.UserData.isZoomOn = true;
        fig.UserData.isZoomOn = true;

        % Turn off pan
        if panBtn.UserData.isPanOn
            pan(ax, 'off');
            panBtn.Text = 'Pan On';
            panBtn.UserData.isPanOn = false;
        end
    else
        zoom(ax, 'off');
        btn.Text = 'Zoom On';
        btn.UserData.isZoomOn = false;
        fig.UserData.isZoomOn = false;
    end
end

function togglePan(ax, btn, fig)
    zoomBtn = btn.UserData.zoomBtn;
    if ~btn.UserData.isPanOn
        pan(ax, 'on');
        btn.Text = 'Pan Off';
        btn.UserData.isPanOn = true;

        % Turn off zoom
        if zoomBtn.UserData.isZoomOn
            zoom(ax, 'off');
            zoomBtn.Text = 'Zoom On';
            zoomBtn.UserData.isZoomOn = false;
            fig.UserData.isZoomOn = false;
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

function handleScrollZoom(ax, fig, event)
    % Only respond if Zoom mode is active
    if ~isfield(fig.UserData, 'isZoomOn') || ~fig.UserData.isZoomOn
        return;
    end

    % Cursor position in axes
    cursor = ax.CurrentPoint(1, 1:2);
    xCenter = cursor(1);
    yCenter = cursor(2);

    % Get current axis limits
    xlim = ax.XLim;
    ylim = ax.YLim;

    % Zoom factor
    zoomFactor = 1.1;
    if event.VerticalScrollCount > 0
        scale = zoomFactor;      % Scroll down → zoom out
    else
        scale = 1 / zoomFactor;  % Scroll up → zoom in
    end

    % Apply zoom centered on cursor
    newXLim = xCenter + (xlim - xCenter) * scale;
    newYLim = yCenter + (ylim - yCenter) * scale;

    ax.XLim = newXLim;
    ax.YLim = newYLim;
end
