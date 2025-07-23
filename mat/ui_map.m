function geoplot_track_ui()
    % Create figure
    fig = uifigure('Name', 'Geo Track Viewer', 'Position', [100, 100, 800, 600]);

    % Create geoaxes
    gax = geoaxes(fig, 'Position', [0.05 0.2 0.9 0.75]);
    geobasemap(gax, 'streets'); % Other options: 'satellite', 'topographic', etc.
    title(gax, 'GPS Track');
    hold(gax, 'on');

    % Generate synthetic GPS track
    numPoints = 100;
    lat = linspace(34.05, 34.15, numPoints) + 0.005*sin(linspace(0, 4*pi, numPoints));
    lon = linspace(-118.25, -118.15, numPoints) + 0.005*cos(linspace(0, 4*pi, numPoints));
    time = linspace(0, 10, numPoints); % Simulated time from 0 to 10 seconds

    % Full track for reference (gray line)
    geoplot(gax, lat, lon, '-', 'Color', [0.8 0.8 0.8]);

    % Partial animated track
    hTrack = geoplot(gax, lat(1), lon(1), 'r-', 'LineWidth', 2);
    hMarker = geoplot(gax, lat(1), lon(1), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

    % Create time slider
    sld = uislider(fig, ...
        'Position', [150, 50, 500, 3], ...
        'Limits', [0, 10], ...
        'MajorTicks', 0:1:10, ...
        'ValueChangedFcn', @(sld, event) updateTrack(sld.Value));

    % Callback to update track
    function updateTrack(currentTime)
        [~, idx] = min(abs(time - currentTime));
        set(hTrack, 'LatitudeData', lat(1:idx), 'LongitudeData', lon(1:idx));
        set(hMarker, 'LatitudeData', lat(idx), 'LongitudeData', lon(idx));
    end
end
